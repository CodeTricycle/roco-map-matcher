"""
洛克王国世界地图 一键爬虫
==========================

一个脚本搞定:
  1. 下载 + 拼接瓦片地图
     - URL 模板从 https://map.17173.com/rocom/maps/shijie 的前端 bundle 解析得到
     - 按 Web Mercator 在 bounds 内枚举瓦片，并发下载到 tiles/{z}/
     - Pillow 拼成 stitched_z{z}.png
  2. 写出 map_info.json （zoom / tile 原点 / 像素尺寸 / bounds）
  3. 从官方接口拉取 marker
     - http://terra-api.17173.com/app/location/list?mapIds=4010
     - 用与拼接图一致的 Web Mercator 公式把 latitude/longitude 直接换算成 x/y 像素，
       覆盖写入 mark.json （写之前自动备份成 mark.json.bak.YYYYmmdd_HHMMSS）

常用:
  python crawler.py                    # 默认 zoom=13，全套跑一遍
  python crawler.py --zoom 11          # 改成 zoom 11
  python crawler.py --skip-tiles       # 只刷新 mark.json
  python crawler.py --skip-marks       # 只下载/拼接瓦片
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

# ---------------------------------------------------------------------------
# 地图配置 —— 这些常量是从前端 bundle 抓到的
# ---------------------------------------------------------------------------
MAP_ID = 4010
GAME_NAME = "rocom"
MAP_VERSION = "v3_7f2d9c"
BOUNDS = [-1.4, 0.0, 0.0, 1.4]           # [west, south, east, north]   (度)
MIN_ZOOM = 9
MAX_ZOOM = 13
TILE_SIZE = 256

TILE_URL_TEMPLATE = (
    "http://ue.17173cdn.com/a/terra/tiles/"
    f"{GAME_NAME}/{MAP_ID}_{MAP_VERSION}"
    "/{z}/{y}_{x}.png?v1"
)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA, "Referer": "https://map.17173.com/"}

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Web Mercator 坐标换算
# ---------------------------------------------------------------------------
def lnglat_to_tile(lng: float, lat: float, z: int) -> tuple[float, float]:
    """(经度, 纬度, zoom) -> (浮点 tile_x, tile_y) —— Mapbox GL 默认 Web Mercator"""
    n = 2 ** z
    x = (lng + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def lnglat_to_pixel(lng: float, lat: float, info: dict) -> tuple[float, float]:
    """换算到拼接大图的像素坐标。"""
    z = info["z"]
    ts = info["tile_size"]
    tx, ty = lnglat_to_tile(lng, lat, z)
    return (tx - info["tx_min"]) * ts, (ty - info["ty_min"]) * ts


def tile_range_for_bounds(bounds: list[float], z: int) -> tuple[int, int, int, int]:
    """返回覆盖 bounds 的 [tx_min, ty_min, tx_max, ty_max]，闭区间。"""
    west, south, east, north = bounds
    x0, y0 = lnglat_to_tile(west, north, z)       # 左上
    x1, y1 = lnglat_to_tile(east, south, z)       # 右下
    tx_min = math.floor(x0)
    ty_min = math.floor(y0)
    tx_max = math.floor(x1 - 1e-9)
    ty_max = math.floor(y1 - 1e-9)
    if tx_max < tx_min:
        tx_max = tx_min
    if ty_max < ty_min:
        ty_max = ty_min
    return tx_min, ty_min, tx_max, ty_max


# ---------------------------------------------------------------------------
# 下载
# ---------------------------------------------------------------------------
def download_tile(session: requests.Session, z: int, x: int, y: int, dest: Path) -> tuple[int, int, bool]:
    if dest.exists() and dest.stat().st_size > 0:
        return x, y, True
    url = TILE_URL_TEMPLATE.format(z=z, x=x, y=y)
    try:
        r = session.get(url, headers=HEADERS, timeout=15)
    except requests.RequestException:
        return x, y, False
    if r.status_code != 200 or not r.content:
        return x, y, False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return x, y, True


def download_all_tiles(z: int, workers: int = 16) -> dict:
    tx_min, ty_min, tx_max, ty_max = tile_range_for_bounds(BOUNDS, z)
    cols = tx_max - tx_min + 1
    rows = ty_max - ty_min + 1
    print(f"[zoom {z}] tile grid: x=[{tx_min}..{tx_max}] y=[{ty_min}..{ty_max}]  => {cols}x{rows} = {cols*rows} tiles")

    tiles_dir = ROOT / "images"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    with requests.Session() as session, ThreadPoolExecutor(max_workers=workers) as pool:
        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                dest = tiles_dir / f"{ty}_{tx}.png"
                jobs.append(pool.submit(download_tile, session, z, tx, ty, dest))

        done = 0
        ok = 0
        total = len(jobs)
        for fut in as_completed(jobs):
            _, _, success = fut.result()
            done += 1
            if success:
                ok += 1
            if done % 25 == 0 or done == total:
                print(f"  downloaded {done}/{total}  (ok={ok})")
    return {
        "z": z,
        "tx_min": tx_min,
        "ty_min": ty_min,
        "tx_max": tx_max,
        "ty_max": ty_max,
        "cols": cols,
        "rows": rows,
        "tiles_dir": str(tiles_dir),
    }


# ---------------------------------------------------------------------------
# 拼接
# ---------------------------------------------------------------------------
def stitch(meta: dict) -> Path:
    z = meta["z"]
    tx_min, ty_min = meta["tx_min"], meta["ty_min"]
    cols, rows = meta["cols"], meta["rows"]
    tiles_dir = Path(meta["tiles_dir"])
    big = Image.new("RGBA", (cols * TILE_SIZE, rows * TILE_SIZE), (0, 0, 0, 0))
    missing = 0
    for row in range(rows):
        for col in range(cols):
            ty = ty_min + row
            tx = tx_min + col
            p = tiles_dir / f"{ty}_{tx}.png"
            if not p.exists():
                missing += 1
                continue
            try:
                img = Image.open(p).convert("RGBA")
            except Exception:
                missing += 1
                continue
            big.paste(img, (col * TILE_SIZE, row * TILE_SIZE))
    out = ROOT / f"big_map.png"
    big.save(out)
    print(f"[stitch] {out.name}  {big.size}  missing={missing}")
    return out


# ---------------------------------------------------------------------------
# 从官方接口拉取 markers 并把 lat/lng 直接换算成像素 x/y
# ---------------------------------------------------------------------------
# 这台 API 的 HTTPS 在 Python ssl 下会触发 UNEXPECTED_EOF_WHILE_READING（curl 没问题），
# 所以走 HTTP，CDN 同样支持。
MARK_API_URL = "http://terra-api.17173.com/app/location/list?mapIds={map_id}"


def fetch_markers_from_api(map_id: int = MAP_ID) -> list[dict]:
    url = MARK_API_URL.format(map_id=map_id)
    print(f"[api] GET {url}")
    r = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=False)
    if r.status_code != 200:
        raise RuntimeError(f"API returned {r.status_code}")
    payload = r.json()
    if not isinstance(payload, dict):
        raise RuntimeError("unexpected response (not a JSON object)")
    if isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload.get("list"), list):
        return payload["list"]
    raise RuntimeError(f"cannot find marker list in payload (keys={list(payload)})")


def normalize_marker(m: dict) -> dict:
    """与前端 axios 拦截器里做的清洗保持一致。"""
    if "map_id" in m and "mapId" not in m:
        m["mapId"] = m["map_id"]
    if isinstance(m.get("image"), str):
        m["image"] = m["image"].replace("http://", "//")
    if isinstance(m.get("images"), list):
        m["images"] = [
            (s.replace("http://", "//") if isinstance(s, str) else s)
            for s in m["images"]
        ]
    try:
        m["latitude"] = float(m.get("latitude"))
        m["longitude"] = float(m.get("longitude"))
    except (TypeError, ValueError):
        pass
    return m


def backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = path.with_suffix(path.suffix + f".bak.{stamp}")
    shutil.copy2(path, dest)
    return dest


def save_marks(map_id: int, info: dict, out_path: Path, round_n: int = 2) -> None:
    """拉取 -> 归一化 -> lat/lng 转 x/y -> 备份旧 mark.json -> 写新文件。"""
    try:
        markers = fetch_markers_from_api(map_id)
    except Exception as e:
        print(f"[api] marker fetch failed: {e}")
        return
    print(f"[api] got {len(markers)} markers")

    w, h = info.get("image_size", [None, None])
    converted = inside = skipped = 0
    for m in markers:
        normalize_marker(m)
        lat = m.get("latitude")
        lng = m.get("longitude")
        if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
            skipped += 1
            continue
        px, py = lnglat_to_pixel(float(lng), float(lat), info)
        m["x"] = round(px, round_n)
        m["y"] = round(py, round_n)
        converted += 1
        if w and h and 0 <= px <= w and 0 <= py <= h:
            inside += 1

    bak = backup_file(out_path)
    if bak:
        print(f"[backup] {out_path.name} -> {bak.name}")
    out_path.write_text(
        json.dumps(markers, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[api] converted={converted}  inside_image={inside}  skipped={skipped}")
    print(f"[api] wrote {out_path.name}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="下载 + 拼接瓦片地图，并从官方接口拉取 marker（自动 lat/lng→x/y）"
    )
    ap.add_argument("--zoom", type=int, default=13,
                    help=f"zoom level ({MIN_ZOOM}..{MAX_ZOOM}), default 13")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-tiles", action="store_true",
                    help="不下载/拼接瓦片，沿用现有 map_info.json (只刷新 mark.json)")
    ap.add_argument("--skip-stitch", action="store_true",
                    help="下载瓦片但不拼大图")
    ap.add_argument("--skip-marks", action="store_true",
                    help="不从接口拉 marker")
    ap.add_argument("--mark-out", default=str(ROOT / "mark.json"))
    ap.add_argument("--round", type=int, default=2,
                    help="像素坐标保留小数位数（默认 2）")
    args = ap.parse_args()

    if not (MIN_ZOOM <= args.zoom <= MAX_ZOOM):
        print(f"zoom {args.zoom} out of range [{MIN_ZOOM}..{MAX_ZOOM}]", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    info_path = ROOT / "map_info.json"

    if args.skip_tiles:
        if not info_path.exists():
            print("--skip-tiles 需要已存在的 map_info.json", file=sys.stderr)
            sys.exit(1)
        meta = json.loads(info_path.read_text(encoding="utf-8"))
        print(f"[skip-tiles] reuse {info_path.name}  zoom={meta.get('z')}")
    else:
        meta = download_all_tiles(args.zoom, workers=args.workers)
        if not args.skip_stitch:
            stitched = stitch(meta)
            meta["stitched"] = str(stitched)
            meta["image_size"] = [meta["cols"] * TILE_SIZE, meta["rows"] * TILE_SIZE]
        meta["bounds"] = BOUNDS
        meta["tile_size"] = TILE_SIZE
        meta["tile_url_template"] = TILE_URL_TEMPLATE
        meta["map_id"] = MAP_ID
        meta["map_version"] = MAP_VERSION
        info_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[info] wrote {info_path.name}")

    if not args.skip_marks:
        save_marks(MAP_ID, meta, Path(args.mark_out), round_n=args.round)

    print(f"[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
