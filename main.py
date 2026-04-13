import sys
import os
import re
import json
import time
import numpy as np
import cv2
import urllib.request
from mss import mss
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QPushButton, QSpinBox,
                              QDoubleSpinBox, QGroupBox, QFormLayout, QCheckBox,
                              QTabWidget, QTreeWidget, QTreeWidgetItem,
                              QSplitter)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint

TILE_SIZE = 256
CONFIG_FILE = "config.json"
IMAGES_DIR = "images"
MARK_DIR = "mark"
CATEGORIES_FILE = "categories.json"
MARK_JSON_FILE = "mark.json"
BIG_MAP_FILE = "big_map.png"
ICONS_DIR = "icons"
X_START = -12
Y_START = -9
ICON_RENDER_SIZE = 20


def _download_icon(url, path):
    try:
        urllib.request.urlretrieve(url, path)
        return True
    except Exception:
        return False


def load_icon(url, size=ICON_RENDER_SIZE):
    if not url:
        return None
    if not os.path.exists(ICONS_DIR):
        os.makedirs(ICONS_DIR)
    fname = url.rsplit('/', 1)[-1]
    local_path = os.path.join(ICONS_DIR, fname)
    if not os.path.exists(local_path):
        if not _download_icon(url, local_path):
            return None
    img = cv2.imread(local_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def overlay_icon(canvas, icon, cx, cy):
    if icon is None:
        return
    h, w = icon.shape[:2]
    x1 = cx - w // 2
    y1 = cy - h // 2
    x2 = x1 + w
    y2 = y1 + h
    if x1 < 0 or y1 < 0 or x2 > canvas.shape[1] or y2 > canvas.shape[0]:
        return
    if icon.shape[2] == 4:
        alpha = icon[:, :, 3].astype(np.float32) / 255.0
        for c in range(3):
            canvas[y1:y2, x1:x2, c] = (
                icon[:, :, c].astype(np.float32) * alpha +
                canvas[y1:y2, x1:x2, c].astype(np.float32) * (1 - alpha)
            ).astype(np.uint8)
    else:
        canvas[y1:y2, x1:x2] = icon


def load_categories(path=CATEGORIES_FILE):
    with open(path, 'r', encoding='utf-8') as f:
        s = f.read()
    s2 = re.sub(r'(?<=[{,\[\s])(\w+)\s*:', r'"\1":', s)
    return json.loads(s2)


def load_mark_json(path=MARK_JSON_FILE):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_mark_json(data, path=MARK_JSON_FILE):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def latlon_to_pixel(lat, lon, img_w, img_h):
    lon_min, lon_max = -1.240031316278176, -0.17679503697396726
    lat_min, lat_max = 0.12232307852095042, 1.1362947897960156
    x = (lon - lon_min) / (lon_max - lon_min) * img_w
    y = (1.0 - (lat - lat_min) / (lat_max - lat_min)) * img_h
    return int(x), int(y)


def pixel_to_latlon(px, py, img_w, img_h):
    lon_min, lon_max = -1.240031316278176, -0.17679503697396726
    lat_min, lat_max = 0.12232307852095042, 1.1362947897960156
    lon = lon_min + px / img_w * (lon_max - lon_min)
    lat = lat_max - py / img_h * (lat_max - lat_min)
    return lat, lon


def split_bitmap(bitmap_path, output_dir, x_start=X_START, y_start=Y_START):
    if not os.path.exists(bitmap_path):
        return
    img = cv2.imread(bitmap_path, cv2.IMREAD_COLOR)
    if img is None:
        return
    h, w = img.shape[:2]
    cols = w // TILE_SIZE
    rows = h // TILE_SIZE
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for row in range(rows):
        for col in range(cols):
            x = x_start + col
            y = y_start + row
            tile = img[row * TILE_SIZE:(row + 1) * TILE_SIZE,
                       col * TILE_SIZE:(col + 1) * TILE_SIZE]
            cv2.imwrite(os.path.join(output_dir, f"{x}_{y}.png"), tile)


class TileManager:
    def __init__(self):
        self.tiles_gray = {}
        self.mark_tiles = {}
        self.keypoints = {}
        self.descriptors = {}
        self.all_descriptors = None
        self.desc_to_tile = []
        self.flann = None
        self.sift = None

    def load(self, nfeatures=1000, progress_cb=None):
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)
        image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.png')]
        total_img = len(image_files)
        for i, f in enumerate(image_files):
            x, y = self._parse_name(f)
            img = cv2.imread(os.path.join(IMAGES_DIR, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.tiles_gray[(x, y)] = img
                kp, des = self.sift.detectAndCompute(img, None)
                self.keypoints[(x, y)] = kp if kp else []
                self.descriptors[(x, y)] = des
            if progress_cb and i % 20 == 0:
                progress_cb(f"加载原图 {i + 1}/{total_img}")

        mark_files = [f for f in os.listdir(MARK_DIR) if f.endswith('.png')]
        total_mark = len(mark_files)
        for i, f in enumerate(mark_files):
            x, y = self._parse_name(f)
            img = cv2.imread(os.path.join(MARK_DIR, f), cv2.IMREAD_COLOR)
            if img is not None:
                self.mark_tiles[(x, y)] = img
            if progress_cb and i % 20 == 0:
                progress_cb(f"加载标注 {i + 1}/{total_mark}")

        all_des = []
        self.desc_to_tile = []
        for key, des in self.descriptors.items():
            if des is not None and len(des) > 0:
                for i in range(len(des)):
                    self.desc_to_tile.append((key, i))
                all_des.append(des)
        if all_des:
            self.all_descriptors = np.vstack(all_des).astype(np.float32)
            flann_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.flann = cv2.FlannBasedMatcher(flann_params, search_params)
            self.flann.add([self.all_descriptors])
            self.flann.train()
            if progress_cb:
                progress_cb(f"FLANN索引构建完成: {len(self.desc_to_tile)}个特征点")

    def _parse_name(self, filename):
        name = filename[:-4]
        idx = name.rfind('_')
        return int(name[:idx]), int(name[idx + 1:])

    def get_mark_composite(self, center_abs_x, center_abs_y, display_size=400):
        half = display_size // 2
        left = center_abs_x - half
        top = center_abs_y - half
        x_min = int(np.floor(left / TILE_SIZE))
        x_max = int(np.floor((left + display_size - 1) / TILE_SIZE))
        y_min = int(np.floor(top / TILE_SIZE))
        y_max = int(np.floor((top + display_size - 1) / TILE_SIZE))
        composite = np.zeros((display_size, display_size, 3), dtype=np.uint8)
        for tx in range(x_min, x_max + 1):
            for ty in range(y_min, y_max + 1):
                if (tx, ty) not in self.mark_tiles:
                    continue
                tile = self.mark_tiles[(tx, ty)]
                tile_abs_left = tx * TILE_SIZE
                tile_abs_top = ty * TILE_SIZE
                overlap_left = max(tile_abs_left, int(left))
                overlap_top = max(tile_abs_top, int(top))
                overlap_right = min(tile_abs_left + TILE_SIZE, int(left + display_size))
                overlap_bottom = min(tile_abs_top + TILE_SIZE, int(top + display_size))
                if overlap_right <= overlap_left or overlap_bottom <= overlap_top:
                    continue
                src_left = overlap_left - tile_abs_left
                src_top = overlap_top - tile_abs_top
                src_right = overlap_right - tile_abs_left
                src_bottom = overlap_bottom - tile_abs_top
                dst_left = overlap_left - int(left)
                dst_top = overlap_top - int(top)
                dst_right = overlap_right - int(left)
                dst_bottom = overlap_bottom - int(top)
                composite[dst_top:dst_bottom, dst_left:dst_right] = \
                    tile[src_top:src_bottom, src_left:src_right]
        return composite


class PositionStabilizer:
    def __init__(self, alpha=0.3, jump_threshold=256, confirm_count=3):
        self.alpha = alpha
        self.jump_threshold = jump_threshold
        self.confirm_count = confirm_count
        self.smooth_x = None
        self.smooth_y = None
        self._pending_x = None
        self._pending_y = None
        self._jump_confirm = 0

    def update(self, cx, cy):
        if self.smooth_x is None:
            self.smooth_x = cx
            self.smooth_y = cy
            return cx, cy
        dist = ((cx - self.smooth_x) ** 2 + (cy - self.smooth_y) ** 2) ** 0.5
        if dist > self.jump_threshold:
            if self._pending_x is not None:
                pending_dist = ((cx - self._pending_x) ** 2 + (cy - self._pending_y) ** 2) ** 0.5
                if pending_dist < self.jump_threshold:
                    self._jump_confirm += 1
                    self._pending_x = cx
                    self._pending_y = cy
                else:
                    self._pending_x = cx
                    self._pending_y = cy
                    self._jump_confirm = 1
            else:
                self._pending_x = cx
                self._pending_y = cy
                self._jump_confirm = 1
            if self._jump_confirm >= self.confirm_count:
                self.smooth_x = self._pending_x
                self.smooth_y = self._pending_y
                self._pending_x = None
                self._pending_y = None
                self._jump_confirm = 0
        else:
            self.smooth_x = self.smooth_x * (1 - self.alpha) + cx * self.alpha
            self.smooth_y = self.smooth_y * (1 - self.alpha) + cy * self.alpha
            self._pending_x = None
            self._pending_y = None
            self._jump_confirm = 0
        return self.smooth_x, self.smooth_y

    def reset(self):
        self.smooth_x = None
        self.smooth_y = None
        self._pending_x = None
        self._pending_y = None
        self._jump_confirm = 0


class MatchThread(QThread):
    result_signal = pyqtSignal(object)
    status_signal = pyqtSignal(str)

    def __init__(self, tile_manager, config):
        super().__init__()
        self.tile_manager = tile_manager
        self.config = config
        self._running = False
        self.sift = cv2.SIFT_create(nfeatures=config.get('sift_nfeatures', 1000))
        self._frame_times = []
        self.stabilizer = PositionStabilizer(alpha=0.3, jump_threshold=TILE_SIZE, confirm_count=3)

    def run(self):
        self._running = True
        with mss() as sct:
            while self._running:
                try:
                    t0 = time.perf_counter()
                    result = self._do_match(sct)
                    t1 = time.perf_counter()
                    self._frame_times.append(t1 - t0)
                    if len(self._frame_times) > 30:
                        self._frame_times.pop(0)
                    fps = len(self._frame_times) / sum(self._frame_times) if sum(self._frame_times) > 0 else 0
                    result['fps'] = fps
                    self.result_signal.emit(result)
                except Exception as e:
                    self.status_signal.emit(f"匹配错误: {e}")

    def _do_match(self, sct):
        monitor = {
            "top": self.config["top"],
            "left": self.config["left"],
            "width": self.config["width"],
            "height": self.config["height"]
        }
        shot = np.array(sct.grab(monitor))
        shot_bgr = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
        shot_gray = cv2.cvtColor(shot_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(shot_gray, None)
        if des is None or len(kp) < 2:
            return {'screenshot': shot_bgr, 'match': None}
        raw_matches = self.tile_manager.flann.knnMatch(des, k=2)
        ratio = self.config.get('ratio_threshold', 0.75)
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio * n.distance:
                    good.append(m)
        min_good = self.config.get('min_good_matches', 3)
        if len(good) < min_good:
            return {'screenshot': shot_bgr, 'match': None}
        tile_abs_positions = {}
        src_pts = []
        dst_pts = []
        for m in good:
            tile_key, local_idx = self.tile_manager.desc_to_tile[m.trainIdx]
            sx, sy = kp[m.queryIdx].pt
            tx, ty = self.tile_manager.keypoints[tile_key][local_idx].pt
            abs_x = tile_key[0] * TILE_SIZE + tx - sx
            abs_y = tile_key[1] * TILE_SIZE + ty - sy
            if tile_key not in tile_abs_positions:
                tile_abs_positions[tile_key] = []
            tile_abs_positions[tile_key].append((abs_x, abs_y))
            src_pts.append([sx, sy])
            dst_pts.append([abs_x, abs_y])
        src_pts = np.float32(src_pts)
        dst_pts = np.float32(dst_pts)
        if len(src_pts) >= 4:
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is not None:
                inlier_mask = mask.ravel().astype(bool)
                inlier_abs_x = dst_pts[inlier_mask, 0]
                inlier_abs_y = dst_pts[inlier_mask, 1]
                if len(inlier_abs_x) >= min_good:
                    med_abs_x = float(np.median(inlier_abs_x))
                    med_abs_y = float(np.median(inlier_abs_y))
                else:
                    all_abs_x = [ax for positions in tile_abs_positions.values() for ax, _ in positions]
                    all_abs_y = [ay for positions in tile_abs_positions.values() for _, ay in positions]
                    med_abs_x = float(np.median(all_abs_x))
                    med_abs_y = float(np.median(all_abs_y))
            else:
                all_abs_x = [ax for positions in tile_abs_positions.values() for ax, _ in positions]
                all_abs_y = [ay for positions in tile_abs_positions.values() for _, ay in positions]
                med_abs_x = float(np.median(all_abs_x))
                med_abs_y = float(np.median(all_abs_y))
        else:
            all_abs_x = [ax for positions in tile_abs_positions.values() for ax, _ in positions]
            all_abs_y = [ay for positions in tile_abs_positions.values() for _, ay in positions]
            med_abs_x = float(np.median(all_abs_x))
            med_abs_y = float(np.median(all_abs_y))
        sw = self.config["width"]
        sh = self.config["height"]
        cx_min = int(np.floor(med_abs_x / TILE_SIZE))
        cx_max = int(np.floor((med_abs_x + sw - 1) / TILE_SIZE))
        cy_min = int(np.floor(med_abs_y / TILE_SIZE))
        cy_max = int(np.floor((med_abs_y + sh - 1) / TILE_SIZE))
        covered = []
        for tx in range(cx_min, cx_max + 1):
            for ty in range(cy_min, cy_max + 1):
                covered.append((tx, ty))
        center_x = med_abs_x + sw / 2.0
        center_y = med_abs_y + sh / 2.0
        smooth_cx, smooth_cy = self.stabilizer.update(center_x, center_y)
        best_tile = max(tile_abs_positions, key=lambda k: len(tile_abs_positions[k]))
        return {
            'screenshot': shot_bgr,
            'match': {
                'abs_pos': (med_abs_x, med_abs_y),
                'center': (smooth_cx, smooth_cy),
                'covered_tiles': covered,
                'best_tile': best_tile,
                'num_good_matches': len(good),
                'tile_match_counts': {k: len(v) for k, v in tile_abs_positions.items()},
            }
        }

    def stop(self):
        self._running = False
        self.stabilizer.reset()


class DisplayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(400, 400)
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.label = QLabel(self)
        self.label.setFixedSize(400, 400)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background-color: black;")
        self.fps_label = QLabel(self)
        self.fps_label.setStyleSheet(
            "color: lime; font-size: 14px; font-weight: bold; background-color: rgba(0,0,0,150); padding: 2px 6px;"
        )
        self.fps_label.move(5, 5)
        self.fps_label.hide()
        self.show_fps = False
        self._drag_pos = None

    def set_fps_visible(self, visible):
        self.show_fps = visible
        if not visible:
            self.fps_label.hide()

    def update_fps(self, fps):
        if self.show_fps:
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.fps_label.show()
            self.fps_label.raise_()

    def update_image(self, cv_img):
        if cv_img is None:
            return
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            400, 400, Qt.AspectRatioMode.KeepAspectRatioByExpanding))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.hide()


class MatchTab(QWidget):
    def __init__(self, tile_manager, config, display_window, parent=None):
        super().__init__(parent)
        self.tile_manager = tile_manager
        self.config = config
        self.display_window = display_window
        self.match_thread = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        ss_group = QGroupBox("截图配置")
        ss_form = QFormLayout()
        self.spin_top = QSpinBox()
        self.spin_top.setRange(0, 10000)
        self.spin_top.setValue(self.config["top"])
        self.spin_left = QSpinBox()
        self.spin_left.setRange(0, 10000)
        self.spin_left.setValue(self.config["left"])
        self.spin_width = QSpinBox()
        self.spin_width.setRange(1, 2000)
        self.spin_width.setValue(self.config["width"])
        self.spin_height = QSpinBox()
        self.spin_height.setRange(1, 2000)
        self.spin_height.setValue(self.config["height"])
        ss_form.addRow("Top:", self.spin_top)
        ss_form.addRow("Left:", self.spin_left)
        ss_form.addRow("Width:", self.spin_width)
        ss_form.addRow("Height:", self.spin_height)
        ss_group.setLayout(ss_form)
        layout.addWidget(ss_group)

        m_group = QGroupBox("匹配配置")
        m_form = QFormLayout()
        self.spin_nfeatures = QSpinBox()
        self.spin_nfeatures.setRange(100, 10000)
        self.spin_nfeatures.setValue(self.config.get("sift_nfeatures", 1000))
        self.spin_ratio = QDoubleSpinBox()
        self.spin_ratio.setRange(0.1, 0.99)
        self.spin_ratio.setSingleStep(0.05)
        self.spin_ratio.setValue(self.config.get("ratio_threshold", 0.75))
        self.spin_min_matches = QSpinBox()
        self.spin_min_matches.setRange(1, 100)
        self.spin_min_matches.setValue(self.config.get("min_good_matches", 3))
        self.chk_fps = QCheckBox("显示帧数")
        self.chk_fps.setChecked(self.config.get("show_fps", False))
        m_form.addRow("SIFT特征数:", self.spin_nfeatures)
        m_form.addRow("比率阈值:", self.spin_ratio)
        m_form.addRow("最小匹配数:", self.spin_min_matches)
        m_form.addRow(self.chk_fps)
        m_group.setLayout(m_form)
        layout.addWidget(m_group)

        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("加载瓦片")
        self.btn_load.clicked.connect(self._on_load)
        self.btn_reload = QPushButton("重载瓦片")
        self.btn_reload.clicked.connect(self._on_reload)
        self.btn_reload.setEnabled(False)
        self.btn_start = QPushButton("开始匹配")
        self.btn_start.clicked.connect(self._on_toggle)
        self.btn_start.setEnabled(False)
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_reload)
        btn_layout.addWidget(self.btn_start)
        layout.addLayout(btn_layout)

        preview_group = QGroupBox("截图预览")
        preview_layout = QVBoxLayout()
        self.preview_label = QLabel("等待截图...")
        self.preview_label.setFixedSize(200, 200)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid gray; background-color: #2b2b2b; color: white;")
        preview_layout.addWidget(self.preview_label, alignment=Qt.AlignmentFlag.AlignCenter)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        self.status_label = QLabel("就绪 - 请先加载瓦片")
        layout.addWidget(self.status_label)
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

    def _on_load(self):
        self.btn_load.setEnabled(False)
        self.status_label.setText("正在加载瓦片并计算SIFT特征...")
        QApplication.processEvents()
        nfeatures = self.spin_nfeatures.value()
        self.tile_manager.load(
            nfeatures=nfeatures,
            progress_cb=lambda msg: (self.status_label.setText(msg), QApplication.processEvents())
        )
        n_img = len(self.tile_manager.tiles_gray)
        n_mark = len(self.tile_manager.mark_tiles)
        n_des = len(self.tile_manager.desc_to_tile)
        self.status_label.setText(f"加载完成: {n_img}原图, {n_mark}标注, {n_des}特征点")
        self.btn_load.setEnabled(True)
        self.btn_reload.setEnabled(True)
        self.btn_start.setEnabled(True)

    def _on_reload(self):
        self.btn_reload.setEnabled(False)
        self.status_label.setText("正在重载瓦片...")
        QApplication.processEvents()
        self.tile_manager.tiles_gray.clear()
        self.tile_manager.mark_tiles.clear()
        self.tile_manager.keypoints.clear()
        self.tile_manager.descriptors.clear()
        self.tile_manager.desc_to_tile.clear()
        nfeatures = self.spin_nfeatures.value()
        self.tile_manager.load(
            nfeatures=nfeatures,
            progress_cb=lambda msg: (self.status_label.setText(msg), QApplication.processEvents())
        )
        n_img = len(self.tile_manager.tiles_gray)
        n_mark = len(self.tile_manager.mark_tiles)
        n_des = len(self.tile_manager.desc_to_tile)
        self.status_label.setText(f"重载完成: {n_img}原图, {n_mark}标注, {n_des}特征点")
        self.btn_reload.setEnabled(True)

    def _on_toggle(self):
        if self.match_thread is not None and self.match_thread.isRunning():
            self.match_thread.stop()
            self.match_thread.wait(3000)
            self.match_thread = None
            self.btn_start.setText("开始匹配")
            self.display_window.hide()
        else:
            self._update_config()
            self.match_thread = MatchThread(self.tile_manager, self.config)
            self.match_thread.result_signal.connect(self._on_result)
            self.match_thread.status_signal.connect(self._on_status)
            self.match_thread.start()
            self.btn_start.setText("停止匹配")
            self.display_window.move(0, 0)
            self.display_window.show()

    def _update_config(self):
        self.config["top"] = self.spin_top.value()
        self.config["left"] = self.spin_left.value()
        self.config["width"] = self.spin_width.value()
        self.config["height"] = self.spin_height.value()
        self.config["sift_nfeatures"] = self.spin_nfeatures.value()
        self.config["ratio_threshold"] = self.spin_ratio.value()
        self.config["min_good_matches"] = self.spin_min_matches.value()
        self.config["show_fps"] = self.chk_fps.isChecked()
        self.display_window.set_fps_visible(self.chk_fps.isChecked())
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def _on_result(self, result):
        fps = result.get('fps', 0)
        self.display_window.update_fps(fps)
        shot = result['screenshot']
        shot_small = cv2.resize(shot, (200, 200))
        rgb = cv2.cvtColor(shot_small, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(qt_img))
        match = result['match']
        if match is None:
            self.info_label.setText("未找到匹配")
            self.status_label.setText("未找到匹配 - 特征点不足")
            return
        center_x, center_y = match['center']
        composite = self.tile_manager.get_mark_composite(center_x, center_y, 400)
        cv2.circle(composite, (200, 200), 5, (0, 0, 255), -1)
        self.display_window.update_image(composite)
        tiles_str = ", ".join(f"({x},{y})" for x, y in match['covered_tiles'])
        match_info = ", ".join(
            f"({k[0]},{k[1]}):{v}" for k, v in sorted(match['tile_match_counts'].items())
        )
        self.info_label.setText(
            f"匹配特征: {match['num_good_matches']} | 覆盖瓦片: {tiles_str}\n"
            f"各瓦片匹配数: {match_info}\n绝对中心: ({center_x:.0f}, {center_y:.0f})"
        )
        self.status_label.setText(
            f"匹配成功 - {match['num_good_matches']}个特征点, {len(match['covered_tiles'])}张瓦片"
        )

    def _on_status(self, msg):
        self.status_label.setText(msg)

    def stop_match(self):
        if self.match_thread is not None and self.match_thread.isRunning():
            self.match_thread.stop()
            self.match_thread.wait(3000)
            self.match_thread = None
            self.btn_start.setText("开始匹配")


class MapLabel(QLabel):
    right_clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            pos = event.position().toPoint()
            self.right_clicked.emit(pos.x(), pos.y())
        super().mousePressEvent(event)


class MapView(QWidget):
    right_clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
        self._pixmap = QPixmap()
        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self._dragging = False
        self._drag_start = QPoint()
        self._offset_start = QPoint()

    def set_pixmap(self, pixmap, initial_scale):
        self._pixmap = pixmap
        self._zoom = initial_scale
        self._offset = QPoint(0, 0)
        self.update()

    def update_pixmap(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def reset_view(self, initial_scale):
        self._zoom = initial_scale
        self._offset = QPoint(0, 0)
        self.update()

    def _img_pos(self, widget_pos):
        return QPoint(
            int((widget_pos.x() - self._offset.x()) / self._zoom),
            int((widget_pos.y() - self._offset.y()) / self._zoom)
        )

    def paintEvent(self, event):
        if self._pixmap.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.translate(self._offset)
        painter.scale(self._zoom, self._zoom)
        painter.drawPixmap(0, 0, self._pixmap)
        painter.end()

    def wheelEvent(self, event):
        if self._pixmap.isNull():
            return
        angle = event.angleDelta().y()
        if angle > 0:
            factor = 1.2
        elif angle < 0:
            factor = 1.0 / 1.2
        else:
            return
        new_zoom = self._zoom * factor
        new_zoom = max(0.05, min(new_zoom, 10.0))
        pos = event.position().toPoint()
        self._offset = QPoint(
            int(pos.x() - (pos.x() - self._offset.x()) * new_zoom / self._zoom),
            int(pos.y() - (pos.y() - self._offset.y()) * new_zoom / self._zoom)
        )
        self._zoom = new_zoom
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start = event.position().toPoint()
            self._offset_start = self._offset
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.RightButton:
            pos = event.position().toPoint()
            img_pos = self._img_pos(pos)
            self.right_clicked.emit(img_pos.x(), img_pos.y())

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = event.position().toPoint() - self._drag_start
            self._offset = self._offset_start + delta
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)


class AnnotateTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.categories = []
        self.marks = []
        self.cat_id_map = {}
        self.cat_icon_map = {}
        self.icon_cache = {}
        self.selected_cat_ids = set()
        self.big_map_img = None
        self.display_scale = 1.0
        self._init_ui()
        self._load_data()

    def _init_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setLayout(QVBoxLayout(self))
        self.layout().addWidget(splitter)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("分类")
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.MultiSelection)
        self.tree.itemClicked.connect(self._on_tree_click)
        left_layout.addWidget(self.tree)
        self.btn_clear_sel = QPushButton("清空选择")
        self.btn_clear_sel.clicked.connect(self._on_clear_selection)
        left_layout.addWidget(self.btn_clear_sel)
        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.map_view = MapView()
        right_layout.addWidget(self.map_view, stretch=1)
        splitter.addWidget(right_widget)

        bottom_layout = QHBoxLayout()
        self.btn_save = QPushButton("保存图片")
        self.btn_save.clicked.connect(self._on_save)
        self.lbl_status = QLabel("")
        bottom_layout.addWidget(self.btn_save)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.lbl_status)
        right_layout.addLayout(bottom_layout)

        splitter.setSizes([200, 800])

    def _load_data(self):
        try:
            self.categories = load_categories()
        except Exception:
            self.categories = []
        self.marks = load_mark_json()
        self._build_tree()
        self._load_big_map()

    def _build_tree(self):
        self.tree.clear()
        self.cat_id_map = {}
        self.cat_icon_map = {}
        for group in self.categories:
            group_item = QTreeWidgetItem(self.tree, [group['title']])
            group_item.setData(0, Qt.ItemDataRole.UserRole, None)
            for cat in group.get('categories', []):
                cat_item = QTreeWidgetItem(group_item, [cat['title']])
                cat_item.setData(0, Qt.ItemDataRole.UserRole, cat['id'])
                self.cat_id_map[cat['id']] = cat['title']
                icon_url = cat.get('icon', '')
                self.cat_icon_map[cat['id']] = icon_url
                if icon_url and cat['id'] not in self.icon_cache:
                    icon = load_icon(icon_url)
                    if icon is not None:
                        self.icon_cache[cat['id']] = icon
            group_item.setExpanded(False)
        self.tree.collapseAll()

    def _on_tree_click(self, item, column):
        self.selected_cat_ids = set()
        for sel_item in self.tree.selectedItems():
            cat_id = sel_item.data(0, Qt.ItemDataRole.UserRole)
            if cat_id is not None:
                self.selected_cat_ids.add(cat_id)
        self._render_map(preserve_view=True)

    def _on_clear_selection(self):
        self.tree.clearSelection()
        self.selected_cat_ids = set()
        self._render_map(preserve_view=True)

    def _load_big_map(self):
        if os.path.exists(BIG_MAP_FILE):
            img = cv2.imread(BIG_MAP_FILE, cv2.IMREAD_COLOR)
            if img is not None:
                self.big_map_img = img
                self._render_map()
                self.lbl_status.setText(f"地图加载: {img.shape[1]}x{img.shape[0]}")
                return
        self.lbl_status.setText("未找到 big_map.png")

    def _render_map(self, preserve_view=False):
        if self.big_map_img is None:
            return

        canvas = self.big_map_img.copy()
        visible_cats = set(self.selected_cat_ids)

        for mark in self.marks:
            cat_id = mark.get('category_id')
            if cat_id not in visible_cats:
                continue
            lat = mark.get('latitude', 0)
            lon = mark.get('longitude', 0)
            px, py = latlon_to_pixel(lat, lon, canvas.shape[1], canvas.shape[0])
            icon = self.icon_cache.get(cat_id)
            if icon is not None:
                overlay_icon(canvas, icon, px, py)
            else:
                cv2.circle(canvas, (px, py), 8, (0, 0, 255), -1)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)

        max_w = 1200
        if w > max_w:
            initial_scale = max_w / w
        else:
            initial_scale = 1.0
        self.display_scale = initial_scale

        if preserve_view:
            self.map_view.update_pixmap(pixmap)
        else:
            self.map_view.set_pixmap(pixmap, initial_scale)

    def _on_save(self):
        if self.big_map_img is None:
            self.lbl_status.setText("未加载地图")
            return

        canvas = self.big_map_img.copy()
        visible_cats = set(self.selected_cat_ids)
        for mark in self.marks:
            cat_id = mark.get('category_id')
            if cat_id not in visible_cats:
                continue
            lat = mark.get('latitude', 0)
            lon = mark.get('longitude', 0)
            px, py = latlon_to_pixel(lat, lon, canvas.shape[1], canvas.shape[0])
            icon = self.icon_cache.get(cat_id)
            if icon is not None:
                overlay_icon(canvas, icon, px, py)
            else:
                cv2.circle(canvas, (px, py), 8, (0, 0, 255), -1)

        if not os.path.exists(MARK_DIR):
            os.makedirs(MARK_DIR)
        h, w = canvas.shape[:2]
        cols = w // TILE_SIZE
        rows = h // TILE_SIZE
        for row in range(rows):
            for col in range(cols):
                x = X_START + col
                y = Y_START + row
                tile = canvas[row * TILE_SIZE:(row + 1) * TILE_SIZE,
                              col * TILE_SIZE:(col + 1) * TILE_SIZE]
                cv2.imwrite(os.path.join(MARK_DIR, f"{x}_{y}.png"), tile)
        self.lbl_status.setText(f"已分割到 {MARK_DIR}/")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("洛克王国地图匹配器")
        self.setMinimumSize(900, 700)
        self.tile_manager = TileManager()
        self.config = self._load_config()
        self.display_window = DisplayWindow()
        self._init_ui()

    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "top": 60, "left": 2293, "width": 223, "height": 223,
            "sift_nfeatures": 1000,
            "ratio_threshold": 0.75, "min_good_matches": 3,
            "show_fps": False
        }

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        self.match_tab = MatchTab(self.tile_manager, self.config, self.display_window)
        self.annotate_tab = AnnotateTab()
        self.tabs.addTab(self.match_tab, "匹配")
        self.tabs.addTab(self.annotate_tab, "标注")
        layout.addWidget(self.tabs)

    def closeEvent(self, event):
        self.match_tab.stop_match()
        self.display_window.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
