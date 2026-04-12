import sys
import os
import json
import time
import numpy as np
import cv2
from mss import mss
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QPushButton, QSpinBox,
                              QDoubleSpinBox, QGroupBox, QFormLayout, QCheckBox)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal

TILE_SIZE = 256
CONFIG_FILE = "config.json"
IMAGES_DIR = "images"
MARK_DIR = "mark"


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
        self.stabilizer = PositionStabilizer(
            alpha=0.3,
            jump_threshold=TILE_SIZE,
            confirm_count=3
        )

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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("洛克王国地图匹配器")
        self.setMinimumWidth(360)
        self.tile_manager = TileManager()
        self.match_thread = None
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

    def _save_config(self):
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

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
        self.btn_start = QPushButton("开始匹配")
        self.btn_start.clicked.connect(self._on_toggle)
        self.btn_start.setEnabled(False)
        btn_layout.addWidget(self.btn_load)
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
        self.btn_start.setEnabled(True)

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
        self._save_config()

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
            f"匹配特征: {match['num_good_matches']} | "
            f"覆盖瓦片: {tiles_str}\n"
            f"各瓦片匹配数: {match_info}\n"
            f"绝对中心: ({center_x:.0f}, {center_y:.0f})"
        )
        self.status_label.setText(
            f"匹配成功 - {match['num_good_matches']}个特征点, "
            f"{len(match['covered_tiles'])}张瓦片"
        )

    def _on_status(self, msg):
        self.status_label.setText(msg)

    def closeEvent(self, event):
        if self.match_thread is not None and self.match_thread.isRunning():
            self.match_thread.stop()
            self.match_thread.wait(3000)
        self.display_window.close()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
