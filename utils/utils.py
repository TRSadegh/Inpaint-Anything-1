import cv2
import numpy as np
from PIL import Image
from typing import Any, Dict, List


def load_img_to_array(img_p):
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)


def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)


def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask

def erode_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.erode(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask

def show_mask(ax, mask: np.ndarray, random_color=False):
    mask = mask.astype(np.uint8)
    if np.max(mask) == 255:
        mask = mask / 255
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_img = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_img)


def show_points(ax, coords: List[List[float]], labels: List[int], size=375):
    coords = np.array(coords)
    labels = np.array(labels)
    color_table = {0: 'red', 1: 'green'}
    for label_value, color in color_table.items():
        points = coords[labels == label_value]
        ax.scatter(points[:, 0], points[:, 1], color=color, marker='*',
                   s=size, edgecolor='white', linewidth=1.25)

def get_clicked_point(img_path):
    img = cv2.imread(img_path)
    original_img = img.copy()
    cv2.namedWindow("Select Point")
    
    # Add instruction text to the image
    instruction_img = img.copy()
    cv2.putText(instruction_img, "Left-click to select point", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(instruction_img, "Right-click to confirm and continue", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(instruction_img, "Press 'q' to quit", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Select Point", instruction_img)
    
    print("Instructions:")
    print("- Left-click to select/update point")
    print("- Right-click to confirm selection and continue")
    print("- Press 'q' to quit")

    last_point = []
    keep_looping = True

    def mouse_callback(event, x, y, flags, param):
        nonlocal last_point, keep_looping, img

        if event == cv2.EVENT_LBUTTONDOWN:
            if last_point:
                cv2.circle(img, tuple(last_point), 5, (0, 0, 0), -1)
            last_point = [x, y]
            cv2.circle(img, tuple(last_point), 5, (0, 0, 255), -1)
            
            # Show updated image with instructions
            display_img = img.copy()
            cv2.putText(display_img, "Right-click to confirm", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Select Point", display_img)
            print(f"Point selected: ({x}, {y}) - Right-click to confirm")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            print("Point confirmed! Continuing...")
            keep_looping = False

    cv2.setMouseCallback("Select Point", mouse_callback)

    while keep_looping:
        key = cv2.waitKey(30) & 0xFF  # Changed from 1 to 30 milliseconds
        if key == ord('q'):
            print("Quitting...")
            keep_looping = False
            last_point = []
            break

    cv2.destroyAllWindows()
    return last_point