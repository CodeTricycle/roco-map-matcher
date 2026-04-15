# 洛克王国世界地图工具集

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.6+-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-orange.svg)](https://opencv.org/)

一个专为《洛克王国：世界》游戏开发的地图工具集，包含地图数据爬取、瓦片拼接、实时位置匹配等功能。

---

## 📦 项目结构

```
roco/
├── crawler.py          # 地图爬虫：下载瓦片、拼接大图、拉取标记数据
├── main.py             # 地图匹配器GUI主程序
├── config.json         # 应用程序配置文件
├── categories.json     # 标记分类数据
├── map_info.json       # 地图元数据（瓦片范围、尺寸等）
├── mark.json           # 游戏标记点数据（魔力之源、宠物位置等）
├── big_map.png         # 拼接后的完整地图
├── requirements.txt    # Python依赖
└── roco.spec           # PyInstaller打包配置
```

---

## 🚀 功能特性

### 1. 地图数据爬取 (`crawler.py`)

- **瓦片下载**：从 17173 CDN 并发下载地图瓦片
- **大图拼接**：使用 Pillow 将瓦片拼接成完整地图
- **标记数据拉取**：从官方 API 获取游戏内标记点（魔力之源、宠物等）
- **坐标转换**：将经纬度自动转换为像素坐标

```bash
# 默认 zoom=13，全套跑一遍
python crawler.py

# 指定 zoom 级别
python crawler.py --zoom 11

# 只刷新标记数据（跳过瓦片下载）
python crawler.py --skip-tiles

# 只下载瓦片（跳过标记拉取）
python crawler.py --skip-marks
```

### 2. 实时地图匹配器 (`main.py`)

基于 PyQt6 开发的 GUI 应用程序，提供两个核心功能：

#### 🎯 匹配模式

- **实时屏幕捕获**：使用 MSS 捕获指定屏幕区域
- **SIFT 特征匹配**：基于 OpenCV SIFT 算法实时匹配游戏画面与地图
- **位置稳定器**：平滑算法防止位置抖动
- **实时预览**：悬浮窗显示当前在地图上的位置

#### 📝 标注模式

- **交互式地图**：支持缩放、拖拽浏览大地图
- **分类筛选**：按类别筛选标记点（魔力之源、宠物、道具等）
- **图标渲染**：自动下载并渲染分类图标
- **瓦片导出**：将带标注的地图导出为瓦片

```bash
# 启动 GUI 应用程序
python main.py
```

---

## 📋 安装要求

### 系统要求

- Windows 10/11
- Python 3.12

### Python 依赖

```bash
pip install -r requirements.txt
```

主要依赖项：

| 包名 | 版本 | 用途 |
|------|------|------|
| PyQt6 | >=6.6.1 | GUI 界面 |
| opencv-python | >=4.8.1 | 图像处理、SIFT 匹配 |
| numpy | >=1.26.2 | 数值计算 |
| mss | >=9.0.1 | 屏幕捕获 |
| requests | - | HTTP 请求 |
| pillow | - | 图像拼接 |

---

## ⚙️ 配置文件

### config.json

```json
{
  "top": 95,
  "left": 2330,
  "width": 150,
  "height": 150,
  "sift_nfeatures": 1000,
  "ratio_threshold": 0.75,
  "min_good_matches": 5,
  "max_fps": 30,
  "show_fps": true
}
```

| 参数 | 说明 |
|------|------|
| `top/left/width/height` | 截图区域坐标和尺寸 |
| `sift_nfeatures` | SIFT 特征点数量 |
| `ratio_threshold` | 匹配比率阈值（Lowe's ratio test）|
| `min_good_matches` | 最小有效匹配数 |
| `max_fps` | 最大帧率限制 |
| `show_fps` | 是否显示 FPS |

---

## 📦 打包可执行文件

使用 PyInstaller 打包为独立可执行文件：

```bash
# 安装 PyInstaller
pip install pyinstaller

# 打包主程序
pyinstaller roco.spec
```

打包后的文件位于 `dist/` 目录下。

---

## 🗺️ 地图数据来源

- 瓦片服务器：`http://ue.17173cdn.com/a/terra/tiles/rocom/`
- 标记 API：`http://terra-api.17173.com/app/location/list`
- 地图网站：https://map.17173.com/rocom/maps/shijie

---

## ⚠️ 免责声明

本项目仅供学习交流使用，不得用于任何商业用途。游戏数据版权归腾讯游戏/洛克王国官方所有。

---

## 📄 许可证

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📞 联系方式

如有问题或建议，请在 GitHub 上提交 Issue。
