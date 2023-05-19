import numpy as np
from imutils import face_utils
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('file_dlib/shape_predictor_68_face_landmarks.dat')
blink_count = 0  # số lần chớp mắt
blink_threshold = 1  # ngưỡng chớp mắt
blink_time = 1  # thời gian chớp mắt
timer_started = False
timer_start = 0
eye_threshold = 0.25
is_sleeping = False


# Tính tỉ lệ mắt
def calculate_eye_ratio(start_point, end_point, landmarks):
    # Tính toán khoảng cách giữa các điểm đặt trên khuôn mặt
    # -------------------------------- p1, p2, p3, p4, p5, p6
    # vị trí mắt phải có các điểm mắt [36, 37, 38, 39, 40, 41]
    # vị trí mắt trái có các điểm mắt [42, 43, 44, 45, 46, 47]
    # A = ||P2 - P6||
    A = np.linalg.norm(landmarks[start_point + 1] - landmarks[end_point])
    # B = ||P3 - P5||
    B = np.linalg.norm(landmarks[start_point + 2] - landmarks[end_point - 1])
    # C = ||P1 - P4||
    C = np.linalg.norm(landmarks[start_point] - landmarks[end_point - 2])
    # Tính tỉ lệ mắt
    eye_ratio = (A + B) / (2 * C)
    return eye_ratio


# kiểm thử video real-time từ camera
# import cv2
#
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)
#     for face in faces:
#         landmarks = predictor(gray, face)
#         landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
#         left_eye_ratio = calculate_eye_ratio(36, 41, landmarks)
#         right_eye_ratio = calculate_eye_ratio(42, 47, landmarks)
#         eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
#         if eye_ratio < 0.25:
#             if not timer_started:
#                 timer_started = True
#                 timer_start = cv2.getTickCount()
#             else:
#                 if (cv2.getTickCount() - timer_start) / cv2.getTickFrequency() > blink_time:
#                     blink_count += 1
#                     timer_started = False
#         else:
#             timer_started = False
#
#         if blink_count >= blink_threshold:
#             print('Sleepy alert!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#             blink_count = 0
#             is_sleeping = True
#         if is_sleeping:
#             if eye_ratio >= eye_threshold:
#                 is_sleeping = False
#                 print("bạn đã thức !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     for (i, rect) in enumerate(faces):
#         # dự đoán và chuyển về mảng
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)
#
#         # vẽ các điểm
#         for (x, y) in shape:
#             cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
#
#     cv2.imshow('frame', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
