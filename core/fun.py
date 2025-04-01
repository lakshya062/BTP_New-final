# core/fun.py

import cv2

# Generate and save a marker from DICT_5X5_100 with ID 10
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
marker_id = 10
marker_size = 200  # pixels
marker_image = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size)
cv2.imwrite(f"marker_{marker_id}.png", marker_image)
