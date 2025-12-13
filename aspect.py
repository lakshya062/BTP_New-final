import cv2
import numpy as np
from screeninfo import get_monitors
import datetime
import time

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

# Calculate position to center the camera feed in the second monitor
x_offset = (screen_width - target_width) // 2
# You can change the vertical offset as needed
y_offset = 300

# Open camera
cap = cv2.VideoCapture(3)

# Create fullscreen window and move it to second monitor
window_name = "Camera Feed"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow(window_name, screen_x, screen_y)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Record the starting time for the progress bar
start_time = time.time()

# Define font properties for text display
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.3
color = (255, 255, 255)  # white
thickness = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to target dimensions
    resized_frame = cv2.resize(frame, (target_width, target_height))

    # Create a black canvas for the entire screen
    canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # Paste the resized camera feed onto the canvas
    canvas[y_offset:y_offset+target_height, x_offset:x_offset+target_width] = resized_frame

    # Prepare text details
    now = datetime.datetime.now()
    date_text = "Date: " + now.strftime("%Y-%m-%d")
    time_text = "Time: " + now.strftime("%H:%M:%S")
    personal_text = "Name = Ayush Sachan, Age = 22, Gender = Male"

    # Define positions for the text details
    # Position text just below the camera feed with proper gaps.
    text_start_y = y_offset + target_height + 40  # start 40px below camera feed
    gap = 60  # gap between each line

    cv2.putText(canvas, date_text, (x_offset, text_start_y), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(canvas, time_text, (x_offset, text_start_y + gap), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(canvas, personal_text, (x_offset, text_start_y + 2 * gap), font, font_scale, color, thickness, cv2.LINE_AA)

    # Draw a progress bar of 1 minute below the text details
    # Progress bar dimensions
    progress_bar_width = 300
    progress_bar_height = 20
    progress_bar_x = (screen_width - progress_bar_width) // 2  # center horizontally
    progress_bar_y = text_start_y + 3 * gap + 20  # some gap below the text

    # Calculate elapsed time and progress fraction
    current_time = time.time()
    elapsed = current_time - start_time
    progress_fraction = min(elapsed / 60.0, 1.0)  # cap at 1.0
    filled_width = int(progress_fraction * progress_bar_width)

    # Draw the progress bar border
    cv2.rectangle(canvas, (progress_bar_x, progress_bar_y),
                  (progress_bar_x + progress_bar_width, progress_bar_y + progress_bar_height),
                  color, 2)
    # Draw the filled portion of the progress bar
    cv2.rectangle(canvas, (progress_bar_x, progress_bar_y),
                  (progress_bar_x + filled_width, progress_bar_y + progress_bar_height),
                  color, -1)

    # Reset progress bar after 1 minute
    if elapsed >= 60:
        start_time = current_time

    # Show the final canvas on the fullscreen window
    cv2.imshow(window_name, canvas)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
