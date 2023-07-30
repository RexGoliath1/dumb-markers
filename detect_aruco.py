import cv2
import os
import argparse
from inspect import getmembers
import numpy as np

# Avaliable Tags:
# 'DICT_4X4_100'
# 'DICT_4X4_1000'
# 'DICT_4X4_250'
# 'DICT_4X4_50' 
# 'DICT_5X5_100'
# 'DICT_5X5_1000'
# 'DICT_5X5_250'
# 'DICT_5X5_50' 
# 'DICT_6X6_100'
# 'DICT_6X6_1000'
# 'DICT_6X6_250'
# 'DICT_6X6_50' 
# 'DICT_7X7_100'
# 'DICT_7X7_1000'
# 'DICT_7X7_250'
# 'DICT_7X7_50' 

BGR_BLUE = (0, 0, 255)
BGR_GREEN = (0, 255, 0)
BGR_RED = (0, 0, 255)
BGR_PURPLE = (255, 0, 255)
VISUALIZE_REJECTED = False

def unpack_corners(corners):
    (tl, tr, br, bl) = corners
    # convert each of the (x, y)-coordinate pairs to integers
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    tl = (int(tl[0]), int(tl[1]))
    return (tl, tr, br, bl)

def draw_corners(image, corners, markerID, line_color):
    (tl, tr, br, bl) = unpack_corners(corners)
    # convert each of the (x, y)-coordinate pairs to integers
    cv2.line(image, tl, tr, line_color, 2)
    cv2.line(image, tr, br, line_color, 2)
    cv2.line(image, br, bl, line_color, 2)
    cv2.line(image, bl, tl, line_color, 2)

    cX = int((tl[0] + br[0]) / 2.0)
    cY = int((tl[1] + br[1]) / 2.0)
    cv2.circle(image, (cX, cY), 5, BGR_RED, -1)
    cv2.putText(image, str(markerID),
        (tl[0], tl[1]), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, line_color, 2)

    return image

def run_aruco(dirpath: str, adict_name: str, visualize: bool, marker_size: float, intrinsics: np.ndarray, dist_coeffs: np.ndarray):
    arucoMembers = dict(getmembers(cv2.aruco))
    adict = arucoMembers[adict_name]
    arucoDict = cv2.aruco.getPredefinedDictionary(adict)
    arucoParams = cv2.aruco.DetectorParameters()
    img_names = os.listdir(dirpath)

    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

    for img_name in img_names:
        image = cv2.imread(os.path.join(dirpath, img_name))
        (all_corners, ids, rejected) = cv2.aruco.detectMarkers(
		    image, arucoDict, parameters=arucoParams)

        if len(all_corners) > 0:
            print("[INFO] detected {} markers for '{}'".format(
                len(all_corners), adict_name))

            retvals = []
            rvecs = []
            tvecs = []

            # loop over the detected ArUCo corners
            for (marker_corners, markerID) in zip(all_corners, ids):
                corners = marker_corners.reshape((4, 2))
                retval, R, t = cv2.solvePnP(marker_points, marker_corners, intrinsics, dist_coeffs, False, cv2.SOLVEPNP_IPPE_SQUARE)
                cv2.drawFrameAxes(image, K, dist_coeffs, R, t,  0.02, 3)
                rvecs.append(R)
                tvecs.append(t)
                retvals.append(retval)
                image = draw_corners(image, corners, markerID, BGR_PURPLE)

            print(f"rvecs: {rvecs}")
            print(f"tvecs: {tvecs}")
            print(f"retvals: {retvals}")

            if VISUALIZE_REJECTED:
                for markerCorner in rejected:
                    corners = markerCorner.reshape((4, 2))
                    image = draw_corners(image, corners, "???", BGR_BLUE)

            if visualize:
                cv2.imshow('img',image)
                cv2.waitKey(1000)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    default_path = os.path.dirname(os.path.realpath(__file__))
    default_path = os.path.join(default_path , "aruco_images")

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", type=str, default=default_path, help="Path to folder containing aruco images for detection")
    ap.add_argument("-k", "--intrinsics", type=str, default="intrinsics.npy",help="Path to calibration matrix (numpy file)")
    ap.add_argument("-n", "--new_intrinsics", type=str, default="new_intrinsics.npy", help="Path to new calibration matrix (numpy file)")
    ap.add_argument("-c", "--dist_coeffs", type=str, default="dist_coeffs.npy", help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-s", "--square_size", type=float, default=0.0205, help="Length of one edge (in meters)")
    ap.add_argument("-a", "--adict_name", type=str, default="DICT_6X6_1000", help="ARUCO Dictionary name. Use --help to see details.")
    ap.add_argument("-v", "--visualize", type=bool, default=True, help="To visualize each checkerboard image")

    test = ap.parse_args()
    args = vars(test)
    dirpath = args['dir']
    adict_name = args['adict_name']
    visualize = args['visualize']
    square_size = args['square_size']
    intrinsics = args['intrinsics']
    new_intrinsics = args['new_intrinsics']
    dist_coeffs = args['dist_coeffs']

    K = np.load(intrinsics)
    K_new = np.load(new_intrinsics)
    d = np.load(dist_coeffs)

    run_aruco(dirpath, adict_name, visualize, square_size, K, d)
