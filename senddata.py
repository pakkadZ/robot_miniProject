import cv2
import numpy as np
import pickle
camera=0
# ---------------- Camera undistort maps ----------------
def build_undistort_maps(index_camera=0, calib_pkl="calibration.pkl"):
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

# ---------------- Perspective transform (camera -> robot) ----------------
camera_points = np.float32([[425,49], [161, 43], [157,433], [415, 430]])
world_points  = np.float32([[369.28, 119.42], [211.82,115.41], [217.60, -116.50], [374.26, -113.94]])
P = cv2.getPerspectiveTransform(camera_points, world_points)

def to_pos_robot(cx, cy, angle_deg):
    cam_pt = np.float32([[[cx, cy]]])   # shape (1,1,2)
    robot_pt = cv2.perspectiveTransform(cam_pt, P)[0, 0]  # (x, y)
    rx, ry = float(robot_pt[0]), float(robot_pt[1])
    r = 90.0 - float(angle_deg)
    return rx, ry, r

# ---------------- HSV bounds ----------------
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

TARGET_KEYS = {"1": "Yellow", "2": "Blue", "3": "Green", "4": "Red"}

def main():
    mapx, mapy, _ = build_undistort_maps(index_camera=camera, calib_pkl="calibration.pkl")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera index 0")

    kernel = np.ones((5, 5), np.uint8)
    min_area = 500
    target_color = "Yellow"

    # เก็บผลล่าสุดไว้ใช้ตอนกด n
    last_result = None  # รูปแบบ: (color_name, rx, ry, r)

    print("[INFO] Press 1:Yellow  2:Blue  3:Green  4:Red   | n:print pose once  | q:quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = undistort_frame(frame, mapx, mapy)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower, upper = COLOR_BOUNDS[target_color]
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        biggest = None
        biggest_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > min_area and area > biggest_area:
                biggest = c
                biggest_area = area

        if biggest is not None:
            rect = cv2.minAreaRect(biggest)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(frame, [box], 0, COLOR_BGR[target_color], 2)

            angle = rect[-1]
            if angle < -45:
                angle = 90 + angle
            angle = float(round(angle, 1))

            M = cv2.moments(biggest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
                cv2.putText(frame, f"{target_color} ({cx},{cy}) ang={angle:.1f}",
                            (cx - 60, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                rx, ry, r = to_pos_robot(cx, cy, angle)
                # อัปเดตผลล่าสุด แต่ "ยังไม่พิมพ์"
                last_result = (target_color, rx, ry, r)
        else:
            last_result = None

        # OSD
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (600, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, f"Target: {target_color} | 1:Y 2:B 3:G 4:R | n:print | q:quit",
                    (14, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Find Color -> Robot Coords (press 'n' to print)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
            target_color = TARGET_KEYS[chr(key)]
            print(f"[TARGET] set to {target_color}")
        elif key == ord('n'):
            if last_result is not None:
                name, rx, ry, r = last_result
                print(f"{name}: {rx:.2f}, {ry:.2f}, {r:.2f}")
            else:
                print("NOT FOUND")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
