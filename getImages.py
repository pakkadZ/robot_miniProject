import cv2
import os

save_dir = r"D:\project_Auttanun\robot_findxy\image"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cap = cv2.VideoCapture(0)
num = 0

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Cannot read frame from camera")
        break

    cv2.imshow('Img', img)
    k = cv2.waitKey(5)

    if k == 27:  
        break
    elif k == ord('s'):  
        file_path = os.path.join(save_dir, f'img{num}.png')
        cv2.imwrite(file_path, img)
        print(f"Image saved: {file_path}")
        num += 1

cap.release()
cv2.destroyAllWindows()
