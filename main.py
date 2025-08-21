import cv2
import math
import numpy as np
import copy
from typing import List

"""
RGB constants
B = 0
G = 1
R = 2

def pixel_color_difference(a: List[int], b: List[int]) -> int:
    return math.sqrt((b[R] - a[R])*(b[R] - a[R]) + 
                     (b[G] - a[G])*(b[G] - a[G]) + 
                     (b[B] - a[B])*(b[B] - a[B]))
"""

if __name__ == "__main__":
    image_path = "./tower.jpg"

    img = cv2.imread(image_path)

    rows = len(img)
    cols = len(img[0])
    seams = []

    # calculate weight for each pixel

    """
    ................
    ................
    ................
    ................
    ................
    ................
    ................
    """

    for x in range(1, cols-1):
        seam = 0
        for y in range(1, rows-1):
            weight = 0
            middel = np.array(img[y][x])
            
            left  = np.array(img[y][x-1])
            right = np.array(img[y][x+1])
            up    = np.array(img[y-1][x])
            down  = np.array(img[y+1][x])

            weight += float(np.linalg.norm(middel - left))
            weight += float(np.linalg.norm(middel - right))
            weight += float(np.linalg.norm(middel - up))
            weight += float(np.linalg.norm(middel - down))

            seam += weight

        seams.append(seam)

    average = sum(seams) / len(seams)
    delta = (max(seams) - min(seams)) / 2

    for x in range(cols-2):
        if seams[x] < average - 5:
            for y in range(rows):
                img[y][x+1] = [0, 0, 255]

                # TODO: lets kinda remove the seams at some point

    cv2.imwrite("carved_tower.jpg", img)

    print("---------------------------------")
    print("[INFO] average:", average)
    print("[INFO] delta:", delta)
    print("---------------------------------")
