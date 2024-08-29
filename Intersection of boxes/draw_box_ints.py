import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def draw_box(vertices):
    for idx in range(len(vertices1)):
        cv2.line(image, vertices[idx], vertices[idx-1], (0, 0, 255), 2)

box1 = [(10, 100), (130, 40), (150, 80), (30, 140)]
box2 = [(80, 50), (200, 50), (200, 110), (80, 110)]

image = 255 * np.ones((600, 900, 3), dtype=np.uint8)
draw_box(vertices1)
draw_box(vertices2)


cv2.imshow("img", image)
cv2.waitKey(0)



