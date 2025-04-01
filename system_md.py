# raspberry_client.py

import socket
import struct
import cv2
import numpy as np
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, filename='raspberry_client.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')


def send_frame(sock, frame):
    """
    Encode and send a frame to the server.
    """
    try:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, encoded = cv2.imencode('.jpg', frame, encode_param)
        if not result:
            logging.error("raspberry_client: Failed to encode frame.")
            return False
        data = encoded.tobytes()
        length = len(data)
        # Pack frame length as 4 bytes in network byte order
        sock.sendall(struct.pack('!I', length) + data)
        return True
    except Exception as e:
        logging.error(f"raspberry_client: Error sending frame: {e}")
        return False


def recv_frame(sock):
    """
    Receive a frame from the server.
    """
    try:
        # Receive 4 bytes indicating the length of the frame
        length_bytes = recvall(sock, 4)
        if not length_bytes:
            logging.warning("raspberry_client: No length bytes received.")
            return None
        frame_length = struct.unpack('!I', length_bytes)[0]
        # Receive the frame data
        frame_data = recvall(sock, frame_length)
        if not frame_data:
            logging.warning("raspberry_client: No frame data received.")
            return None
        # Decode the frame
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logging.error(f"raspberry_client: Error receiving frame: {e}")
        return None


def recvall(sock, count):
    """
    Helper function to receive exactly 'count' bytes from the socket.
    """
    buf = b''
    while count > 0:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def main():
    MAIN_SYSTEM_IP = "192.168.1.100"  # Replace with actual main system IP
    CAMERA_INDEX = 0  # Typically 0 for the first camera
    PORT = 8000 + CAMERA_INDEX  # Ensure it matches the main system's FrameServer port

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((MAIN_SYSTEM_IP, PORT))
        logging.info(f"Connected to main system at {MAIN_SYSTEM_IP}:{PORT}.")
    except Exception as e:
        logging.critical(f"raspberry_client: Could not connect to main system: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logging.critical(f"raspberry_client: Could not open camera {CAMERA_INDEX}.")
        sock.close()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Smart Mirror Display", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Smart Mirror Display", 960, 540)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("raspberry_client: Failed to read frame from camera.")
                break

            # Send the frame to the main system
            send_success = send_frame(sock, frame)
            if not send_success:
                logging.warning("raspberry_client: Failed to send frame.")
                break

            # Receive the processed frame from the main system
            processed_frame = recv_frame(sock)
            if processed_frame is None:
                logging.warning("raspberry_client: Failed to receive processed frame.")
                break

            # Display the processed frame
            cv2.imshow("Smart Mirror Display", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("raspberry_client: 'q' pressed. Exiting.")
                break

            time.sleep(0.033)  # Approximately 30 FPS
    except KeyboardInterrupt:
        logging.info("raspberry_client: KeyboardInterrupt received. Exiting.")
    except Exception as e:
        logging.error(f"raspberry_client: Exception occurred: {e}")
    finally:
        cap.release()
        sock.close()
        cv2.destroyAllWindows()
        logging.info("raspberry_client: Closed connections and released resources.")

if __name__ == "__main__":
    main()

[Unit]
Description=Smart Gym Raspberry Pi Client
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/pi/smart_gym/raspberry_client.py
WorkingDirectory=/home/pi/smart_gym/
StandardOutput=inherit
StandardError=inherit
Restart=always
User=pi

[Install]
WantedBy=multi-user.target