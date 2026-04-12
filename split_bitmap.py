import os
import cv2
import numpy as np

TILE_SIZE = 256

X_START = -12
Y_START = -9


def split_bitmap(bitmap_path, output_dir, x_start=X_START, y_start=Y_START):
    if not os.path.exists(bitmap_path):
        print(f"错误: 找不到 {bitmap_path}")
        return

    img = cv2.imread(bitmap_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"错误: 无法读取 {bitmap_path}")
        return

    h, w = img.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    cols = w // TILE_SIZE
    rows = h // TILE_SIZE
    print(f"可分割瓦片: {cols}列 x {rows}行 = {cols * rows}张")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    for row in range(rows):
        for col in range(cols):
            x = x_start + col
            y = y_start + row
            tile = img[row * TILE_SIZE:(row + 1) * TILE_SIZE, col * TILE_SIZE:(col + 1) * TILE_SIZE]
            filename = f"{x}_{y}.png"
            cv2.imwrite(os.path.join(output_dir, filename), tile)
            count += 1

    print(f"完成: 共分割 {count} 张瓦片到 {output_dir}/")


if __name__ == '__main__':
    split_bitmap("big_map_mark.png","mark")
    split_bitmap("big_map.png","images")
