import io
import os
import scipy.misc
import numpy as np
import six
import time
import glob
from IPython.display import display
from official.vision.serving import detection
from send_telegram import send_telegram
from six import BytesIO
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import cv2
import datetime
import threading
from imutils import face_utils
import dlib
import blink_detection as blink


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# load file landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('file_dlib/shape_predictor_68_face_landmarks.dat')

# Load model
tf.keras.backend.clear_session()
model = tf.saved_model.load("export_model/saved_model")
category_index = label_map_util.create_category_index_from_labelmap("label_map.txt", use_display_name=True)

# --------------- detect with image
image_path = 'file_test/74.png'
image_np = load_image_into_numpy_array(image_path)

print("Done load image ")
image_np = cv2.resize(image_np, dsize=None, fx=1.5, fy=1.5)

output_dict = run_inference_for_single_image(model, image_np)
print("Done inference:")
vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    instance_masks=output_dict.get('detection_masks_reframed', None),
    use_normalized_coordinates=True,
    line_thickness=8)
im_width = 864
im_height = 480

ymin, xmin, ymax, xmax = output_dict['detection_boxes'][0]
(left, right, top, bottom) = (round(xmin * im_width), round(xmax * im_width), round(ymin * im_height), round(ymax * im_height))
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
yellow = (0, 255, 255)
# tọa độ bouding box mới
# khoản pixel cộng thêm
pixcel_plus = 40
left_top = (left + pixcel_plus, top + pixcel_plus)
left_bottom = (left + pixcel_plus, bottom - pixcel_plus)
right_top = (right - pixcel_plus, top + pixcel_plus)
right_bottom = (right - pixcel_plus, bottom - pixcel_plus)
# vẽ chấm tròn góc trên bên trái
cv2.circle(image_np, left_top, 5, red, -1)
# vẽ chấm tròn góc dưới bên trái
cv2.circle(image_np, left_bottom, 5, green, -1)
# vẽ đường chéo từ góc trên bên trái tới góc dưới bên phải của bouding box
cv2.line(image_np, left_top, right_bottom, (0, 0, 255), 2)
# vẽ chấm tròn góc trên bên phải bouding box
cv2.circle(image_np, right_top, 5, yellow, -1)
# vẽ chấm tròn góc dưới bên phải bouding box
cv2.circle(image_np, right_bottom, 5, blue, -1)

# vẽ đường chéo từ góc dưới bên trái tới góc trên bên phải của bouding box
cv2.line(image_np, left_bottom, right_top, (0, 0, 255), 2)
# # tính vị trí điểm giữa phía trên của bouding box
# center_top = round((right - left) / 2)
# # tính vị trí điểm giữa bên trái bouding box
# center_bottom = round((bottom - top) / 2)
# # tính toán tọa độ điểm giữa của bouding box
# centroid_x = center_top + left  # toạ độ x
# centroid_y = center_bottom + top  # tọa độ y
# # vẽ toạ độ lên bouding box
# cv2.circle(image_np, (centroid_x, centroid_y), 5, yellow, -1)
print("Done draw on image ")
# cv2.imwrite("file_test/alert.png", cv2.resize(image_np, dsize=None, fx=1, fy=1))
# send_telegram()
while True:
    cv2.imshow('Detect Oject', image_np)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

# ----------------------- Detect with video
#
# video = cv2.VideoCapture("file_test/video1.mp4")
# while True:
#     # Lấy một frame
#     ret, frame = video.read()
#     if not ret:
#         break
#     # Chuyển đổi frame sang numpy array
#     image_np = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     im_height, im_width, chanel = np.shape(image_np)
#     # detection landmark
#     faces = detector(image_np)
#     # Dự đoán bằng model và numpy array
#     output_dict = run_inference_for_single_image(model, image_np)
#     print("Done inference")
#     vis_util.visualize_boxes_and_labels_on_image_array(
#         image_np,
#         output_dict['detection_boxes'],
#         output_dict['detection_classes'],
#         output_dict['detection_scores'],
#         category_index,
#         instance_masks=output_dict.get('detection_masks_reframed', None),
#         use_normalized_coordinates=True,
#         line_thickness=8)
#     print("Done draw on image ", output_dict['detection_boxes'])
#     ymin, xmin, ymax, xmax = output_dict['detection_boxes'][0]
#     (left, right, top, bottom) = (round(xmin * im_width), round(xmax * im_width), round(ymin * im_height), round(ymax * im_height))
#     center_top = round((right - left) / 2)
#     center_bottom = round((bottom -top)/2)
#     centroid_x = center_top + left
#     centroid_y = center_bottom + top
#     cv2.circle(image_np, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
#     # lấy name của output_dict
#     name = category_index[output_dict['detection_classes'][0]]['name']
#     if name == 'lie':
#         print("true")
#         for face in faces:
#             landmarks = predictor(image_np, face)
#             landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
#             left_eye_ratio = blink.calculate_eye_ratio(36, 41, landmarks)
#             right_eye_ratio = blink.calculate_eye_ratio(42, 47, landmarks)
#             eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
#             if eye_ratio < 0.25:
#                 if not blink.timer_started:
#                     blink.timer_started = True
#                     blink.timer_start = cv2.getTickCount()
#                 else:
#                     # tính thời gian thực thi
#                     if (cv2.getTickCount() - blink.timer_start) / cv2.getTickFrequency() > blink.blink_time:
#                         blink.blink_count += 1
#                         blink.timer_started = False
#             else:
#                 timer_started = False
#
#             if blink.blink_count >= blink.blink_threshold:
#                 print('Sleepy alert!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#                 blink.blink_count = 0
#         for (i, rect) in enumerate(faces):
#             # dự đoán và chuyển về mảng
#             shape = predictor(image_np, rect)
#             shape = face_utils.shape_to_np(shape)
#
#             # vẽ các điểm
#             for (x, y) in shape:
#                 cv2.circle(image_np, (x, y), 2, (0, 255, 0), -1)
#     else:
#         pass
#     # Hiển thị kết quả
#     cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
#     # Break the loop if 'q' key is pressed
#     if cv2.waitKey(int(1000 / 60)) & 0xFF == ord('q'):
#         break
#
# video.release()
# cv2.destroyAllWindows()
