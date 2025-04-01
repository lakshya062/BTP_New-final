import cv2
import numpy as np
from screeninfo import get_monitors

# --- Configuration ---
# target_width = 1300  # width for 32-inch view
# target_height = 800  # height for 32-inch view
target_width = 1000
target_height = 1500

# Get monitor info
monitors = get_monitors()
if len(monitors) < 2:
    raise RuntimeError("At least two monitors are required.")

# Assuming the second monitor is the one we want
second_monitor = monitors[1]
screen_width = second_monitor.width
screen_height = second_monitor.height
screen_x = second_monitor.x
screen_y = second_monitor.y

# Calculate position to center the 32-inch video window in 70-inch screen
x_offset = (screen_width - target_width) // 2
# y_offset = (screen_height - target_height) // 2
y_offset = 300

# Open camera
cap = cv2.VideoCapture(2)

# Create fullscreen window and move it to second monitor
window_name = "Camera Feed"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(window_name, screen_x, screen_y)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to 1280x720
    resized_frame = cv2.resize(frame, (target_width, target_height))

    # Create a black canvas and paste resized_frame in center
    canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    canvas[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = resized_frame

    # Show in fullscreen on second monitor
    cv2.imshow(window_name, canvas)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
