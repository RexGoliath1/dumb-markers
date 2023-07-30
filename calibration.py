#!python
"""
python calibration.py --dir calibration/ --square_size 0.024
"""

import numpy as np
import cv2
import os
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(dir_path, "calibration_images")


def calibrate(dirpath: str, square_size: float,  width: int, height: int, visualize: bool = False):
    """ Do some dumb calibration.
            path: path to checkerboard images
            square_size: Square size in meters
            width: checkerboard width, number of inner corners
            height: checkerboard height, number of inner corners            
            visualize: visualize calibration
    """
    epsilon = 0.001
    maxCount = 30
    termCriteria = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = (termCriteria, maxCount, epsilon)

    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    img_names = os.listdir(dirpath)

    for img_name in img_names:
        image = cv2.imread(os.path.join(dirpath, img_name))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            image = cv2.drawChessboardCorners(image, (width, height), corners2, ret)

        if visualize:
            cv2.imshow('img',image)
            cv2.waitKey(1)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

    if visualize:
        for img_name in img_names:
            image = cv2.imread(os.path.join(dirpath, img_name))
            dst = cv2.undistort(image, mtx, dist, None, newcameramtx)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            cv2.imshow('img',image)
            cv2.waitKey(100)

    cv2.destroyAllWindows()

    return [ret, mtx, dist, rvecs, tvecs, newcameramtx]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", help="Path to folder containing checkerboard images for calibration", default=save_path)
    ap.add_argument("-w", "--width", type=int, help="Width of checkerboard (default=9)",  default=8)
    ap.add_argument("-t", "--height", type=int, help="Height of checkerboard (default=6)", default=6)
    ap.add_argument("-s", "--square_size", type=float, default=0.0235, help="Length of one edge (in meters)")
    ap.add_argument("-v", "--visualize", type=bool, default=True, help="To visualize each checkerboard image")
    args = vars(ap.parse_args())
    
    dirpath = args['dir']
    square_size = args['square_size']
    width = args['width']
    height = args['height']
    visualize = args['visualize']

    ret, mtx, dist, rvecs, tvecs, newcameramtx = calibrate(dirpath, square_size, visualize=visualize, width=width, height=height)

    print(mtx)
    print(dist)
    print(newcameramtx)

    np.save("intrinsics", mtx)
    np.save("dist_coeffs", dist)
    np.save("new_intrinsics", newcameramtx)
