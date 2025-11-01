import cv2
import numpy as np
import pickle
import socket
from threading import Thread, Lock
import time
import sys

# ========= CONFIG =========
CALIB_FILE     = "calibration.pkl"     
CAMERA_INDEX   = 0                     # พอร์ตกล้อง
IP_ROBOT       = "192.168.1.6"
PORT           = 6601
MIN_AREA       = 500                  
DRAW_WINDOW    = True             

# ========= SHARED STATE =========
status = 'wait'                        # 'wait' | 'find' | 'find_pos'
pos_frame = None                       # (cx, cy, angle_deg, color_name)
state_lock = Lock()               

# ========= CAMERA UNDISTORT =========
def build_undistort_maps(index_camera=0, calib_pkl=CALIB_FILE):

    with open(calib_pkl, "rb") as f:
        cameraMatrix, distCoeffs = pickle.load(f)

    cap = cv2.VideoCapture(index_camera)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot open camera to build undistort maps.")
    h, w = frame.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, newK, (w, h), cv2.CV_32FC1)
    cap.release()
    return mapx, mapy, (w, h)

def undistort_frame(frame, mapx, mapy):
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

# ========= PERSPECTIVE TRANSFORM (camera -> robot/world) =========

camera_points = np.float32([[189,47], [460, 446], [458,37], [192, 445]])
world_points  = np.float32([[383.65 , -96.80], [223.31 , 138.99], [226.35 , -100.46], [381.16 , 139.24]])
P = cv2.getPerspectiveTransform(camera_points, world_points)

def to_pos_robot(cx, cy, angle_deg):
    cam_pt = np.float32([[[cx, cy]]])     
    robot_pt = cv2.perspectiveTransform(cam_pt, P)[0, 0]  
    rx, ry = float(robot_pt[0]), float(robot_pt[1])
    r = 90.0 - float(angle_deg)
    return rx, ry, r

# ========= HSV BOUNDS & COLORS =========
B_bounds = (np.array([86, 157, 157]),  np.array([158, 255, 255]))
G_bounds = (np.array([43, 104, 142]),  np.array([82, 200, 231]))
Y_bounds = (np.array([24, 108, 143]),  np.array([52, 196, 229]))
R_bounds = (np.array([0, 103, 120]),   np.array([26, 184, 196]))  

COLOR_BOUNDS = {
    "Yellow": Y_bounds,
    "Blue":   B_bounds,
    "Green":  G_bounds,
    "Red":    R_bounds
}
COLOR_BGR = {
    "Yellow": (0, 255, 255),
    "Blue":   (255, 0, 0),
    "Green":  (0, 255, 0),
    "Red":    (0, 0, 255)
}
# ========= THREAD: Robot Communication =========
class Mg400(Thread):
    """สื่อสารกับหุ่นผ่าน TCP socket ตามโปรโตคอล: hi / start / pos?"""
    def __init__(self, ip, port):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"[SOCK] connecting to {ip}:{port}")
        self.sock.connect((ip, port))
        self.sock.settimeout(1.0) 
        self.send_text('hi')
        print("[SOCK] hi sent")
        self.start()

    def send_text(self, text):
        try:
            self.sock.send(text.encode())
        except Exception as e:
            print(f"[SOCK] send error: {e}")

    def recv_bytes(self, maxlen=64):
        try:
            return self.sock.recv(maxlen)
        except socket.timeout:
            return b''
        except Exception as e:
            print(f"[SOCK] recv error: {e}")
            return b''

    def run(self):
        global status, pos_frame
        while True:
            # 1) รอคำสั่งจากหุ่นเมื่อ status == 'wait'
            if status == 'wait':
                data = self.recv_bytes(50)
                if data:
                    if data == b'start':
                        status = 'find'
                        print("[SOCK] -> status=find")
                        time.sleep(0.1)
                    elif data == b'pos?':
                        status = 'find_pos'
                        print("[SOCK] -> status=find_pos")
                        time.sleep(0.1)

            # 2) โหมดค้นหา: ถ้าพบวัตถุ (pos_frame มีค่า) ส่ง 'found'
            if status == 'find':
                while True:
                    with state_lock:
                        cur = pos_frame
                    if cur is not None:
                        self.send_text('found')
                        print("[SOCK] found")
                        status = 'wait'
                        break
                    else:
                        print("[SOCK] Not Found!!")
                    time.sleep(0.1)

            # 3) โหมดขอตำแหน่ง: ส่งพิกัดจริงและมุม
            if status == 'find_pos':
                with state_lock:
                    cur = pos_frame
                if cur is not None:
                    cx, cy, angle_deg, color_name = cur
                    rx, ry, r = to_pos_robot(cx, cy, angle_deg)
                    msg = f'{rx:.2f},{ry:.2f},{r:.2f}'
                    self.send_text(msg)
                    print(f"[SOCK] {color_name}: {msg}")
                    status = 'wait'
                else:
                    print("[SOCK] Not found -> finish")
                    self.send_text('finish')
                    status = 'wait'

            time.sleep(0.05)

# ========= THREAD: Vision Processing =========
class VisionProcessing(Thread):
    """อ่านภาพ, undistort, แยกสี, หาคอนทัวร์ใหญ่สุด พร้อมวาดผล/อัพเดต pos_frame"""
    def __init__(self, camera_index, mapx, mapy, draw_window=True):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")
        self.mapx = mapx
        self.mapy = mapy
        self.kernel = np.ones((5, 5), np.uint8)
        
        self.boxes = ['Red'] 
        #self.boxes = ['Yellow', 'Blue', 'Green', 'Red'] 

        self.draw_window = draw_window
        self.start()

    def run(self):
        global pos_frame

        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("[VISION] camera read failed")
                break

            frame = undistort_frame(frame, self.mapx, self.mapy)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            best = None           # (area, cx, cy, angle_deg, color_name, box_pts)
            best_box = None

            for color_name in self.boxes:
                lower, upper = COLOR_BOUNDS[color_name]
                mask = cv2.inRange(hsv, lower, upper)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    area = cv2.contourArea(c)
                    if area < MIN_AREA:
                        continue

                    rect = cv2.minAreaRect(c)
                    angle = rect[-1]
                    if angle < -45:
                        angle = 90 + angle
                    angle = float(round(angle, 1))

                    M = cv2.moments(c)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    box = cv2.boxPoints(rect).astype(np.int32)

                    # เก็บตัวที่ "ใหญ่สุด"
                    if (best is None) or (area > best[0]):
                        best = (area, cx, cy, angle, color_name, box)
                        best_box = box

            # อัปเดตผลล่าสุดให้ Mg400 ใช้
            with state_lock:
                if best is not None:
                    _, cx, cy, angle, color_name, box = best
                    pos_frame = (cx, cy, angle, color_name)
                else:
                    pos_frame = None

            if self.draw_window:
                if best is not None:
                    _, cx, cy, angle, color_name, box = best
                    cv2.drawContours(frame, [best_box], 0, COLOR_BGR[color_name], 2)
                    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
                    cv2.putText(frame, f"{color_name} ({cx},{cy}) ang={angle:.1f}",
                                (cx - 60, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (520, 60), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                with state_lock:
                    txt = "FOUND" if pos_frame is not None else "NOT FOUND"
                cv2.putText(frame, f"Status: {status} | Vision: {txt} | q:quit",
                            (14, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.imshow("Find Color -> Robot Coords", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            time.sleep(0.01)

        self.cap.release()
        if self.draw_window:
            cv2.destroyAllWindows()

# ========= MAIN =========
def main():
    # 1) เตรียม undistort maps จากไฟล์คาลิเบรต
    try:
        mapx, mapy, _ = build_undistort_maps(index_camera=CAMERA_INDEX, calib_pkl=CALIB_FILE)
    except Exception as e:
        print(f"[INIT] build_undistort_maps failed: {e}")
        sys.exit(1)

    # 2) สตาร์ทเธรดสื่อสารหุ่น
    try:
        mg400_thread = Mg400(IP_ROBOT, PORT)
    except Exception as e:
        print(f"[INIT] socket connect failed: {e}")
        sys.exit(1)

    # 3) สตาร์ทเธรดประมวลผลภาพ
    try:
        vision_thread = VisionProcessing(CAMERA_INDEX, mapx, mapy, draw_window=DRAW_WINDOW)
    except Exception as e:
        print(f"[INIT] vision init failed: {e}")
        sys.exit(1)

    # 4) วนรันจนกด Ctrl+C
    print("[INFO] Running... protocol: hi -> (start/find) -> (pos?)")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Exiting...")

if __name__ == "__main__":
    main()
