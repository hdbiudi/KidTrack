import cv2
import numpy as np
from imutils.video import VideoStream

video = cv2.VideoCapture("file_test/video1.mp4")
# Chua cac diem nguoi dung chon de tao da giac
points = []


def handle_left_click(event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def draw_polygon(frame, points):
    for point in points:
        frame = cv2.circle(frame, (point[0], point[1]), 5, (0, 0, 255), -1)

    frame = cv2.polylines(frame, [np.int32(points)], False, (255, 0, 0), thickness=2)
    return frame


while True:
    ret, frame = video.read()
    if not ret:
        break
    # Ve ploygon
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = draw_polygon(frame, points)
    key = cv2.waitKey(1)
    if cv2.waitKey(int(1000 / 60)) & 0xFF == ord('q'):
        break
    elif key == ord('d'):
        points.append(points[0])

    # Hien anh ra man hinh
    cv2.imshow("Intrusion Warning", frame)

    cv2.setMouseCallback("Intrusion Warning", handle_left_click, points)

video.release()
cv2.destroyAllWindows()
