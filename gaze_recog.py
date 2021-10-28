import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

develope_mode = True

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
print('open cam ...')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print('success open cam ...')
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            landmark_z_list = [results.multi_face_landmarks[0].landmark[zz].z for zz in range(468)]
            landmark_x_list = [results.multi_face_landmarks[0].landmark[xx].x for xx in range(468)]
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
            state = 'focus'
            if right_cheek_diff>left_thres:
                state = 'left'
            elif left_cheek_diff>right_thres:
                state = 'right'
            elif nor_vertical_diff<down_thres:
                state = 'down'
            elif nor_vertical_diff>up_thres:
                state = 'up'

            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks)
            # Flip the image horizontally for a selfie-view display.
            result_img = cv2.flip(image, 1)
            if develope_mode:
                result_img = cv2.putText(result_img, 'up & down:'+'%.3f'%nor_vertical_diff, (10, 60), cv2.FONT_HERSHEY_PLAIN,1, (0, 0, 255), 1, cv2.LINE_AA)
                result_img = cv2.putText(result_img, 'left:'+'%.3f'%right_cheek_diff, (10, 80), cv2.FONT_HERSHEY_PLAIN,1, (0, 0, 255), 1, cv2.LINE_AA)
                result_img = cv2.putText(result_img, 'right:'+'%.3f'%left_cheek_diff, (10, 100), cv2.FONT_HERSHEY_PLAIN,1, (0, 0, 255), 1, cv2.LINE_AA)
            result_img = cv2.putText(result_img, state, (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('MediaPipe Face Mesh', result_img)
        else:
            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()