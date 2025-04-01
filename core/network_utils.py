# # core/network_utils.py

# import socket
# import struct
# import cv2
# import numpy as np
# import threading
# import logging

# logging.basicConfig(level=logging.INFO, filename='network_utils.log',
#                     format='%(asctime)s - %(levelname)s - %(message)s')


# class FrameServer(threading.Thread):
#     """
#     A frame server that listens for a connection from a Raspberry Pi client,
#     receives raw frames, and sends processed frames back.
#     """

#     def __init__(self, pi_id, ip, port):
#         super().__init__()
#         self.pi_id = pi_id
#         self.ip = ip
#         self.port = port
#         self.server_socket = None
#         self.conn = None
#         self.addr = None
#         self.running = True
#         self.frame_received = False
#         self.lock = threading.Lock()
#         self.received_frame = None
#         self.processed_frame = None
#         self.command_queue = []
#         self.command_lock = threading.Lock()

#     def run(self):
#         try:
#             self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             self.server_socket.bind((self.ip, self.port))
#             self.server_socket.listen(1)
#             logging.info(f"FrameServer [{self.pi_id}]: Waiting for connection on {self.ip}:{self.port}...")
#             self.conn, self.addr = self.server_socket.accept()
#             logging.info(f"FrameServer [{self.pi_id}]: Connection from {self.addr} established.")

#             while self.running:
#                 # Receive message type
#                 msg_type = self.recvall(1)
#                 if not msg_type:
#                     logging.warning(f"FrameServer [{self.pi_id}]: No message type received. Closing connection.")
#                     break

#                 msg_type = msg_type.decode('utf-8')

#                 # Receive message length
#                 length_bytes = self.recvall(4)
#                 if not length_bytes:
#                     logging.warning(f"FrameServer [{self.pi_id}]: No message length received. Closing connection.")
#                     break

#                 msg_length = struct.unpack('!I', length_bytes)[0]

#                 # Receive the actual message
#                 message = self.recvall(msg_length)
#                 if not message:
#                     logging.warning(f"FrameServer [{self.pi_id}]: No message received. Closing connection.")
#                     break

#                 if msg_type == 'D':  # Data frame
#                     frame = cv2.imdecode(np.frombuffer(message, dtype=np.uint8), cv2.IMREAD_COLOR)
#                     with self.lock:
#                         self.received_frame = frame
#                         self.frame_received = True
#                 elif msg_type == 'C':  # Command
#                     command = message.decode('utf-8')
#                     logging.info(f"FrameServer [{self.pi_id}]: Received command: {command}")
#                     with self.command_lock:
#                         self.command_queue.append(command)
#                 else:
#                     logging.error(f"FrameServer [{self.pi_id}]: Unknown message type: {msg_type}")

#         except Exception as e:
#             logging.error(f"FrameServer [{self.pi_id}]: Exception occurred: {e}")
#         finally:
#             self.close()

#     def recvall(self, count):
#         buf = b''
#         while count > 0:
#             newbuf = self.conn.recv(count)
#             if not newbuf:
#                 return None
#             buf += newbuf
#             count -= len(newbuf)
#         return buf

#     def send_frame(self, frame):
#         try:
#             # Encode the frame as JPEG
#             encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
#             result, encoded = cv2.imencode('.jpg', frame, encode_param)
#             if not result:
#                 logging.error(f"FrameServer [{self.pi_id}]: Failed to encode frame.")
#                 return False
#             data = encoded.tobytes()
#             # Send as 'D' type
#             self.conn.sendall(b'D' + struct.pack('!I', len(data)) + data)
#             return True
#         except Exception as e:
#             logging.error(f"FrameServer [{self.pi_id}]: Error sending frame: {e}")
#             return False

#     def send_command(self, command):
#         try:
#             data = command.encode('utf-8')
#             self.conn.sendall(b'C' + struct.pack('!I', len(data)) + data)
#             logging.info(f"FrameServer [{self.pi_id}]: Sent command: {command}")
#             return True
#         except Exception as e:
#             logging.error(f"FrameServer [{self.pi_id}]: Error sending command: {e}")
#             return False

#     def get_frame(self):
#         with self.lock:
#             if self.frame_received:
#                 frame = self.received_frame.copy()
#                 self.frame_received = False
#                 return frame
#             else:
#                 return None

#     def get_command(self):
#         with self.command_lock:
#             if self.command_queue:
#                 return self.command_queue.pop(0)
#             else:
#                 return None

#     def close(self):
#         try:
#             if self.conn:
#                 self.conn.close()
#                 logging.info(f"FrameServer [{self.pi_id}]: Connection closed.")
#             if self.server_socket:
#                 self.server_socket.close()
#                 logging.info(f"FrameServer [{self.pi_id}]: Server socket closed.")
#         except Exception as e:
#             logging.error(f"FrameServer [{self.pi_id}]: Error during closing: {e}")

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

def get_page_load_time(url):
    chrome_options = Options()
    

    # Create the driver
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Start timing before page load
        start_time = time.time()

        # Load the page
        driver.get(url)

        # Option 1: Use the Performance Timing API (Legacy but widely supported)
        # JS snippet to get load time in ms
        load_time_script = """
        var performance = window.performance || window.webkitPerformance || window.mozPerformance || window.msPerformance;
        if (performance) {
            var timing = performance.timing;
            return timing.loadEventEnd - timing.navigationStart;
        }
        return 0;
        """
        load_time_ms = driver.execute_script(load_time_script)


        print(f"Load time reported by Performance Timing API: {load_time_ms} ms")

        end_time = time.time()
        measured_time = (end_time - start_time) * 1000
        print(f"Measured load time from Python perspective: {measured_time:.2f} ms")

        return load_time_ms

    finally:
        driver.quit()

if __name__ == "__main__":
    url_to_test = "https://www.example.org"
    get_page_load_time(url_to_test)












