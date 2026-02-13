import numpy as np


def colormap():
    color_list = np.array([
        [255, 0, 255],
        [31, 119, 180],   # 🔵 
        [255, 127, 14],   # 🟠 
        [44, 160, 44],    # 🟢 
        [214, 39, 40],    # 🔴 
    ], dtype=np.uint8)     

    return color_list
