#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2
import numpy as np

from utils import CvFpsCalc
from face_mesh.face_mesh import FaceMesh
from iris_landmark.iris_landmark import IrisLandmark


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--max_num_faces", type=int, default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.7)

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    max_num_faces = args.max_num_faces
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    cap = cv2.VideoCapture(cap_device, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    face_mesh = FaceMesh(
        max_num_faces,
        min_detection_confidence,
        min_tracking_confidence,
    )
    iris_detector = IrisLandmark()

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        display_fps = cvFpsCalc.get()

        ret, image = cap.read()
        if not ret:
            break
        image = cv2.flip(image, 1)
        debug_image = copy.deepcopy(image)
        face_results,origin_result = face_mesh(image)
        for face_result in face_results:
            left_eye, right_eye = face_mesh.calc_around_eye_bbox(face_result)

            left_iris, right_iris = detect_iris(image, iris_detector, left_eye,
                                                right_eye)

            left_center, left_radius = calc_min_enc_losingCircle(left_iris)
            right_center, right_radius = calc_min_enc_losingCircle(right_iris)

            debug_image = draw_debug_image(
                debug_image,
                left_iris,
                right_iris,
                left_center,
                left_radius,
                right_center,
                right_radius,
            )
        if origin_result.multi_face_landmarks:
            landmark_z_list = [origin_result.multi_face_landmarks[0].landmark[zz].z for zz in range(468)]
            landmark_x_list = [origin_result.multi_face_landmarks[0].landmark[xx].x for xx in range(468)]
            # up and down
            # print('forehead:',landmark_z_list[151])
            forehead_relative = landmark_z_list[151]-landmark_z_list[0]
            # print('forehead_relative:',forehead_relative)
            # print('check:',landmark_z_list[175])
            check_relative = landmark_z_list[175]-landmark_z_list[0]
            # print('check_relative:',check_relative)
            vertical_diff = forehead_relative-check_relative
            # print('diff = ', vertical_diff)
            max_vertical_diff = abs(max(landmark_z_list) - min(landmark_z_list))
            # print('max_vertical_diff = ',max_vertical_diff)
            nor_vertical_diff = vertical_diff/max_vertical_diff
            # print('nor_vertical_diff = ',nor_vertical_diff)
            # left and right
            nose = landmark_x_list[1]
            right_cheek = landmark_x_list[93]
            left_cheek = landmark_x_list[323]
            right_cheek_diff = abs(right_cheek-nose)/abs(max(landmark_x_list)-min(landmark_x_list))
            left_cheek_diff = abs(left_cheek-nose)/abs(max(landmark_x_list)-min(landmark_x_list))
            # print('right_cheek_diff : ', right_cheek_diff)
            # print('left_cheek_diff : ', left_cheek_diff)
            

            # recongnition
            up_thres = 0.25
            down_thres = -0.45
            left_thres = 0.65
            right_thres = 0.65
            state = 'front'
            if right_cheek_diff>left_thres:
                state = 'right'
            elif left_cheek_diff>right_thres:
                state = 'left'
            elif nor_vertical_diff<down_thres:
                state = 'down'
            elif nor_vertical_diff>up_thres:
                state = 'up'


            # gaze recog
            print('left_center : ', left_center)
            print('right_center : ', right_center)
            w,h = image.shape[1], image.shape[0]
            left_eye_out = [origin_result.multi_face_landmarks[0].landmark[263].x*w, origin_result.multi_face_landmarks[0].landmark[263].y*h]
            left_eye_in = [origin_result.multi_face_landmarks[0].landmark[362].x*w, origin_result.multi_face_landmarks[0].landmark[362].y*h]
            left_eye_up = [origin_result.multi_face_landmarks[0].landmark[386].x*w, origin_result.multi_face_landmarks[0].landmark[386].y*h]
            left_eye_down = [origin_result.multi_face_landmarks[0].landmark[374].x*w,origin_result.multi_face_landmarks[0].landmark[374].y*h]
            right_eye_out = [origin_result.multi_face_landmarks[0].landmark[33].x*w,origin_result.multi_face_landmarks[0].landmark[33].y*h]
            right_eye_in = [origin_result.multi_face_landmarks[0].landmark[133].x*w,origin_result.multi_face_landmarks[0].landmark[133].y*h]
            right_eye_up = [origin_result.multi_face_landmarks[0].landmark[159].x*w,origin_result.multi_face_landmarks[0].landmark[159].y*h]
            right_eye_down = [origin_result.multi_face_landmarks[0].landmark[145].x*w,origin_result.multi_face_landmarks[0].landmark[145].y*h]
            cv2.circle(debug_image, (int(left_eye_out[0]), int(left_eye_out[1])), 1, (0, 255, 0), 1) #ok
            cv2.circle(debug_image, (int(left_eye_in[0]), int(left_eye_in[1])), 1, (0, 255, 0), 1)#ok
            cv2.circle(debug_image, (int(left_eye_up[0]), int(left_eye_up[1])), 1, (0, 255, 0), 1)#ok
            cv2.circle(debug_image, (int(left_eye_down[0]), int(left_eye_down[1])), 1, (0, 255, 0), 1)#ok
            cv2.circle(debug_image, (int(right_eye_out[0]), int(right_eye_out[1])), 1, (0, 255, 0), 1)#ok
            cv2.circle(debug_image, (int(right_eye_in[0]), int(right_eye_in[1])), 1, (0, 255, 0), 1)#ok
            cv2.circle(debug_image, (int(right_eye_up[0]), int(right_eye_up[1])), 1, (0, 255, 0), 1)#ok
            cv2.circle(debug_image, (int(right_eye_down[0]), int(right_eye_down[1])), 1, (0, 255, 0), 1)#ok
            print('left_eye_out:', left_eye_out)
            print('left_eye_in:', left_eye_in)
            print('left_eye_up:', left_eye_up)
            print('left_eye_down:', left_eye_down)
            print('right_eye_out:', right_eye_out)
            print('right_eye_in:', right_eye_in)
            print('right_eye_up:', right_eye_up)
            print('right_eye_down:', right_eye_down)
            right_eye_horizental_ratio = abs((left_center[0]-right_eye_out[0])/(right_eye_in[0]-right_eye_out[0]))
            left_eye_horizental_ratio = abs((right_center[0]-left_eye_out[0])/(left_eye_in[0]-left_eye_out[0]))
            right_left_horizental_ratio = right_eye_horizental_ratio/left_eye_horizental_ratio
            right_eye_vertical_ratio = abs((right_eye_down[1]-left_center[1])/(right_eye_down[1]-right_eye_up[1]))
            left_eye_vertical_ratio = abs((left_eye_down[1]-right_center[1])/(left_eye_down[1]-left_eye_up[1]))
            vertical_ratio = (right_eye_vertical_ratio+left_eye_vertical_ratio)/2
            eye_up_thres = 0.6
            eye_down_thres = 0.4
            eye_left_thres = 0.75
            eye_right_thres = 1.5
            eye_state = 'gaze'
            if right_left_horizental_ratio<eye_left_thres:
                eye_state = 'left'
            elif right_left_horizental_ratio>eye_right_thres:
                eye_state = 'right'
            elif vertical_ratio<eye_down_thres:
                eye_state = 'down'
            elif vertical_ratio>eye_up_thres:
                eye_state = 'up'
            cv2.putText(debug_image, "FPS:" + str(display_fps), (10, 20),cv2.FONT_HERSHEY_PLAIN,1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(debug_image, "face state:"+state, (10, 40),cv2.FONT_HERSHEY_PLAIN,1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(debug_image, "eye state:"+eye_state, (10, 60),cv2.FONT_HERSHEY_PLAIN,1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(debug_image, "R&L ratio:"+'%.3f'%right_left_horizental_ratio, (10, 80),cv2.FONT_HERSHEY_PLAIN,1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(debug_image, "U&D ratio:"+'%.3f'%vertical_ratio, (10, 100),cv2.FONT_HERSHEY_PLAIN,1, (0, 0, 255), 1, cv2.LINE_AA)

        key = cv2.waitKey(1)
        if key == 27: 
            break


        cv2.imshow('Iris(tflite) Demo', debug_image)

    cap.release()
    cv2.destroyAllWindows()

    return


def detect_iris(image, iris_detector, left_eye, right_eye):
    image_width, image_height = image.shape[1], image.shape[0]
    input_shape = iris_detector.get_input_shape()


    left_eye_x1 = max(left_eye[0], 0)
    left_eye_y1 = max(left_eye[1], 0)
    left_eye_x2 = min(left_eye[2], image_width)
    left_eye_y2 = min(left_eye[3], image_height)
    left_eye_image = copy.deepcopy(image[left_eye_y1:left_eye_y2,
                                         left_eye_x1:left_eye_x2])

    eye_contour, iris = iris_detector(left_eye_image)

    left_iris = calc_iris_point(left_eye, eye_contour, iris, input_shape)


    right_eye_x1 = max(right_eye[0], 0)
    right_eye_y1 = max(right_eye[1], 0)
    right_eye_x2 = min(right_eye[2], image_width)
    right_eye_y2 = min(right_eye[3], image_height)
    right_eye_image = copy.deepcopy(image[right_eye_y1:right_eye_y2,
                                          right_eye_x1:right_eye_x2])

    eye_contour, iris = iris_detector(right_eye_image)

    right_iris = calc_iris_point(right_eye, eye_contour, iris, input_shape)

    return left_iris, right_iris


def calc_iris_point(eye_bbox, eye_contour, iris, input_shape):
    iris_list = []
    for index in range(5):
        point_x = int(iris[index * 3] *
                      ((eye_bbox[2] - eye_bbox[0]) / input_shape[0]))
        point_y = int(iris[index * 3 + 1] *
                      ((eye_bbox[3] - eye_bbox[1]) / input_shape[1]))
        point_x += eye_bbox[0]
        point_y += eye_bbox[1]

        iris_list.append((point_x, point_y))

    return iris_list


def calc_min_enc_losingCircle(landmark_list):
    center, radius = cv2.minEnclosingCircle(np.array(landmark_list))
    center = (int(center[0]), int(center[1]))
    radius = int(radius)

    return center, radius


def draw_debug_image(
    debug_image,
    left_iris,
    right_iris,
    left_center,
    left_radius,
    right_center,
    right_radius,
):

    for point in left_iris:
        cv2.circle(debug_image, (point[0], point[1]), 1, (0, 0, 255), 1)
    for point in right_iris:
        cv2.circle(debug_image, (point[0], point[1]), 1, (0, 0, 255), 1)
    return debug_image


if __name__ == '__main__':
    main()
