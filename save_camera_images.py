"""
    Test script to take images, mostly for calibration images.
"""

import cv2
import os
import time
import argparse

def save_camera_images(dirpath: str, max_images: int = 100):
    """ A dumb saver """
    os.makedirs(dirpath, exist_ok=True)
    cap = cv2.VideoCapture(0)
    num_images = 0
    time.sleep(3)

    for _ in range(0, 1000):
        print("Saving calibration frame...")
        rv, img = cap.read()

        if rv:
            img_path = os.path.join(dirpath, f"{num_images}.png")
            cv2.imwrite(img_path, img)
            print(f"Saved {img_path}")
            num_images += 1
        else:
            print("No image data")

        if num_images >= max_images:
            print("Saved images.")
            break

    cap.release()

if __name__ == '__main__':
    default_path = os.path.dirname(os.path.realpath(__file__))
    default_path = os.path.join(default_path , "aruco_images")

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", type=str, default=default_path, help="Path to folder to save images")
    ap.add_argument("-n", "--num_images", default=100, help="number of images to take ")
    args = vars(ap.parse_args())
    dirpath = args['dir']
    save_camera_images(dirpath)

