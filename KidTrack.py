import cv2
import numpy as np
from Object_Detection import SSDMobileNet
from imutils.video import VideoStream


def handle_left_click(event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


def draw_polygon(frame, points):
    for point in points:
        frame = cv2.circle(frame, (point[0], point[1]), 5, (0, 0, 255), -1)

    frame = cv2.polylines(frame, [np.int32(points)], False, (255, 0, 0), thickness=2)
    return frame


model = SSDMobileNet(detect_class=['sit', 'lie', 'stand'])
detect = False
points = []
video = cv2.VideoCapture("file_test/test_lie.mp4")
# url = 'http://192.168.43.233:4747/video'
# video = cv2.VideoCapture(url)
while True:
    ret, frame = video.read()
    if not ret:
        break
    # Ve ploygon
    frame = cv2.resize(frame, (640, 480))
    frame = draw_polygon(frame, points)
    if detect:
        frame = model.detect(frame=frame, points=points)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key == ord('d'):
        points.append(points[0])
        detect = True
    # Hien anh ra man hinh
    cv2.imshow("KidTrack", frame)
    cv2.setMouseCallback("KidTrack", handle_left_click, points)
video.release()
cv2.destroyAllWindows()
