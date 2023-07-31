Testing out aruco code pose estimation. Default test image folders included.

1. Run save_camera_images on aruco codes and checkerboard calibration.
    - Default sample boards included.
2. Run calibration on camera (default 8x6 checkerboard w/ subpixel estimation + camera mtx refinement)
    - Saves numpy files locally.
2. Run detect_aruco
    - Displays various overlays and prints out distances and orientations.
