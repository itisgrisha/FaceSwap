#!/usr/bin/env python

import cv2
import numpy as np
import time


def get_head_pose(image_points, im, debug=True):
    size = im.shape
    _t = time.time()

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corne
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner

    ])

    # Camera internals

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    print("Camera Matrix :\n {0}".format(camera_matrix))

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points,
                                                                  image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    print("Rotation Vector:\n {0}".format(rotation_vector))
    print("Translation Vector:\n {0}".format(translation_vector))

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

    (nose_end_point2D, jacobian) = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    print("Total time:{}".format(time.time() - _t))
    print(p1, p2)

    # Display image
    if debug:
        for p in image_points:
            cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
        cv2.line(im, p1, p2, (255, 0, 0), 2)
        cv2.imwrite('head_pose_debug_{}.png'.format(time.time()), im)


# Read Image
if __name__ == '__main__':
    import dlib
    from face_points_detection import face_points_detection
    detector = dlib.get_frontal_face_detector()
    PREDICTOR_PATH = '../FaceSwap/models/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    def select_face(im, detector, predictor, r=10):
        fx = max(im.shape)
        fx = int(fx / 200 + 0.5)

        im = cv2.resize(cv2.GaussianBlur(im, (fx + (1-fx % 2), fx + (1-fx % 2)), 0),
                        (0, 0), fx=1/fx, fy=1/fx, interpolation=cv2.INTER_NEAREST)
        faces = detector(im, 1)
        areas = [face.area() for face in faces]
        if len(areas) == 0:
            return np.zeros((68, 2), dtype=np.int)
        bbox = faces[np.argmax(areas)]

        bbox = dlib.rectangle(bbox.left()*fx, bbox.top()*fx, bbox.right()*fx, bbox.bottom()*fx)

        points = np.asarray(face_points_detection(im, bbox, predictor))

        return points

    def draw_landmarks(image, landmarks):
        image = image.copy()
        for point in landmarks:
            cv2.circle(image, tuple(point), 5, (255, 0, 255))
        return image

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = frame[..., ::-1]
        if not ret:
            break
        frame = draw_landmarks(frame, select_face(frame, detector, predictor))
        cv2.imshow('landmarks', frame[..., ::-1])
        q = cv2.waitKey(1) & 0xff
        if q == ord('q'):
            break
        # get_head_pose()
    cv2.destroyAllWindows()

# im = cv2.imread("headPose.jpg")
#
# # 2D image points. If you change the image, you need to change vector
# image_points = np.array([
#     (359, 391),     # 30 Nose tip
#     (399, 561),     # 8 Chin
#     (337, 297),     # 36 Left eye left corner
#     (513, 301),     # 45 Right eye right corne
#     (345, 465),     # 48 Left Mouth corner
#     (453, 469)      # 54 Right mouth corner
# ], dtype="double")
#
#
# get_head_pose(image_points, im, True)
