# core/fun.py

import cv2


def generate_marker(marker_id=10, marker_size=200, output_path=None):
    """Generate and save an ArUco marker image."""
    output_path = output_path or f"marker_{marker_id}.png"
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    marker_image = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size)
    return cv2.imwrite(output_path, marker_image)


if __name__ == "__main__":
    generate_marker()
