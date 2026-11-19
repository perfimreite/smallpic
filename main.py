import sys
import cv2
import math
import numpy as np
import numpy.typing as npt
import copy
import time
from typing import List

# B = 0
# G = 1
# R = 2

# def pixel_color_diff(a: npt.NDArray[np.uint8], b: npt.NDArray[np.uint8]) -> float:
#     # return np.linalg.norm(a - b)
#     return math.sqrt((b[R] - a[R])**2 +
#                      (b[G] - a[G])**2 +
#                      (b[B] - a[B])**2)

def get_density_map(img, rows, cols):
    ret = np.zeros((rows, cols), dtype=np.float32)

    for y in range (1, rows - 1):
        for x in range (1, cols - 1):
            if y > 0 and x > 0:               ret[y][x] += np.linalg.norm(img[y][x] - img[y-1][x-1])
            if y > 0:                         ret[y][x] += np.linalg.norm(img[y][x] - img[y-1][x  ])
            if y > 0 and x < cols - 1:        ret[y][x] += np.linalg.norm(img[y][x] - img[y-1][x+1])
            if x > 0:                         ret[y][x] += np.linalg.norm(img[y][x] - img[y  ][x-1])
            if x < cols - 1:                  ret[y][x] += np.linalg.norm(img[y][x] - img[y  ][x+1])
            if y < rows - 1 and x > 0:        ret[y][x] += np.linalg.norm(img[y][x] - img[y+1][x-1])
            if y < rows - 1:                  ret[y][x] += np.linalg.norm(img[y][x] - img[y+1][x  ])
            if y < rows - 1 and x < cols - 1: ret[y][x] += np.linalg.norm(img[y][x] - img[y+1][x+1])

    return ret

remove_count = 200

def remove_seams(img, rows, cols, density_map):
    for i in range(remove_count):
        print(f"    {i}/{remove_count}", end="\r")

        cum = np.zeros((rows, cols), dtype=np.float32)
        cum[0] = density_map[0]

        for y in range(1, rows):
            for x in range(cols):
                cum[y, x] = density_map[y, x] + min(
                    cum[y-1, x-1] if x > 0 else math.inf,
                    cum[y-1, x],
                    cum[y-1, x+1] if x < cols-1 else math.inf
                )

        index = np.argmin(cum[rows-1])
        carved = [index]

        for y in range(rows-1, 0, -1):
            start = max(0, index - 1)
            end = min(cols, index + 2)

            index = start + np.argmin(cum[y-1, start:end])
            carved.append(index)

        carved.reverse()

        mask = np.ones((rows, cols), dtype=bool)
        mask[np.arange(rows), carved] = False

        img = img[mask].reshape(rows, cols - 1, 3)
        density_map = density_map[mask].reshape(rows, cols - 1)

        cols -= 1

in_path      = "tower.jpg"
out_path     = "carved_tower.jpg"
density_path = "density_map.jpg"

def main():
    start = time.monotonic()

    print(f"Reading '{in_path}'...")
    img = cv2.imread(in_path)
    if img is None:
        print(f"ERROR: Could not read image: {in_path}")
        sys.exit(1)

    rows = len(img)
    cols = len(img[0])

    print("Computing density map...")
    density_map = get_density_map(img.astype(np.float32), rows, cols)

    print("Gathering average density...")
    avg = 0
    for y in range(rows):
        for x in range(cols):
            avg += density_map[y][x]
    avg /= rows * cols

    print(f"Creating '{density_path}'...")
    imgcpy = img.copy()
    for y in range(rows):
        for x in range(cols):
            if density_map[y][x] > avg:
                imgcpy[y][x] = [0, 0, 255 * density_map[y][x] / math.sqrt(3*(255**2))]

    ok = cv2.imwrite(density_path, imgcpy)
    if not ok:
        print(f"ERROR: Failed to create {density_path}")
        sys.exit(1)

    print("Removing seams...")
    img = remove_seams(img, rows, cols, density_map)

    print(f"Creating '{density_path}'...")
    ok = cv2.imwrite(out_path, img)
    if not ok:
        print(f"ERROR: Failed to create {out_path}")
        sys.exit(1)

    print(f"Elapsed time: {round(time.monotonic() - start, 2)}s")

if __name__ == "__main__":
    main()

