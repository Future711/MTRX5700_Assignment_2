"""Estimate camera intrinsic parameters from checkerboard images.

Run this script with a folder containing checkerboard images. It detects chess
corners, calibrates the camera, and stores the intrinsics and reprojection
error in a JSON file.
"""

import os
import argparse
import json
import numpy as np
import cv2

def load_images_from_folder(folder):
    """Load all readable images from a folder.

    Args:
        folder: Directory containing image files.

    Returns:
        List of images loaded with OpenCV in BGR format.
    """
    images = []
    for filename in os.listdir(folder):
        print(os.path.join(folder,filename))
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def main():
    """Run camera intrinsic calibration and write the result to JSON.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description="Calibrate image intrinsics from a collection of checkerboard images.")
    parser.add_argument("image_dir", help="Image directory.")
    parser.add_argument("--output", default="calibration_example.json", help="Output JSON file (default: calibration_example.json).")
    args = parser.parse_args()

    image_dir = args.image_dir

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Object points describe the known checkerboard vertex locations in 3D.
    # Update these parameters to match the physical board being used.
    checkerboard_width = 8
    checkerboard_height = 11
    checkerboard_size = 0.0185
    objp = np.zeros((checkerboard_width*checkerboard_height, 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard_height, 0:checkerboard_width].T.reshape(-1,2)*checkerboard_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = load_images_from_folder(image_dir)

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (checkerboard_height, checkerboard_width), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            # Search in the grayscale image for the corners, refined sub pixel coorindates
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners for introspection - do the corners/vertices drawn back onto the image match with the checkerboard in the camera view?
            cv2.drawChessboardCorners(img, (checkerboard_height, checkerboard_width), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Now re-project images to pixels for re-projection error 
    # use cv2.projectPoints:
    # use computed rvecs, tvecs, mtx, and dist 
    # to reproject known 3D points back into each image, then compute distance (norm) between reprojected points and originally detected imgpoints. Average across all images
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2) # norming is calculating the Euclidean distance 
        total_error += error
    mean_error = total_error / len(objpoints)
    
    print(mtx)
    print(dist)
    print(f"Mean reprojection error (pixels): {mean_error}")

    # Load existing JSON if it exists, otherwise start fresh
    output_path = args.output
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            calib = json.load(f)
    else:
        calib = {}

    calib["K"] = mtx.tolist()
    calib["distortion"] = dist.flatten().tolist()
    calib["reprojection_error"] = mean_error

    with open(output_path, "w") as f:
        json.dump(calib, f, indent=4)

    print(f"Intrinsics saved to {output_path}")

# Example Intrinsic: 
# [[519.26845842   0.         331.11197675]
#  [  0.         518.89359517 229.43433605]
#  [  0.           0.           1.        ]]
# Example Distortion: 
# [[ 0.11418155  0.19343114 -0.00268067  0.00371577 -1.09539701]]


if __name__ == '__main__':
    main()

"""Output given input parameters: 
Input: 
    checkerboard_width = 8
    checkerboard_height = 11
    checkerboard_size = 0.019

Output: 
Intrinsic Matrix: 
[[505.17402857   0.         322.61498844]
 [  0.         504.48645729 235.83829198]
 [  0.           0.           1.        ]]

 Distortion Coefficients
[[ 0.19455044 -0.41789654 -0.00078207  0.00051684  0.1866632 ]]


Own Bag Data: 
Input: 
    checkerboard_width = 8
    checkerboard_height = 11
    checkerboard_size = 0.0185

Output: 
Intrinsic Matrix: 
[[502.31211754   0.         319.10316849]
 [  0.         501.37224574 228.78647846]
 [  0.           0.           1.        ]]

Distortion Coefficients:
[[ 0.18969769 -0.4867124  -0.00198649 -0.00203937  0.28592385]]

Mean reprojection error (pixels): 0.013323200337964282
"""
