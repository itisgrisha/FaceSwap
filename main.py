import sys
import os
import cv2
# import dlib
import argparse
import json

import face_alignment as fca

import numpy as np

from face_points_detection import face_points_detection
from face_swap import warp_image_2d, warp_image_3d, mask_from_points, apply_mask, correct_colours, transformation_from_points
from time import time

# print('LOL')

def select_face_dl(im, fa, r=10):
    points = fa.get_landmarks(im)
    if points is None:
        return None, None, None
    points = points[0]

    im_w, im_h = im.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)

    x, y = int(max(0, left-r)), int(max(0, top-r))
    w, h = int(min(right+r, im_h)-x), int(min(bottom+r, im_w)-y)
    
#     print(x, y, h, w)

    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y+h, x:x+w]


def select_face(im, detector, predictor, r=10):
    fx = max(im.shape)
    fx = int(fx / 200 + 0.5)

    kek_img = cv2.resize(cv2.GaussianBlur(im, (fx + (1-fx % 2), fx + (1-fx % 2)), 0),
                         (0, 0), fx=1/fx, fy=1/fx, interpolation=cv2.INTER_NEAREST)
    faces = face_detection(kek_img, detector)
    areas = [face.area() for face in faces]
    bbox = faces[np.argmax(areas)]

    bbox = dlib.rectangle(bbox.left()*fx, bbox.top()*fx, bbox.right()*fx, bbox.bottom()*fx)

    points = np.asarray(face_points_detection(im, bbox, predictor))

    im_w, im_h = im.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)

    x, y = max(0, left-r), max(0, top-r)
    w, h = min(right+r, im_h)-x, min(bottom+r, im_w)-y

    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y+h, x:x+w]


class Meme():
    def __init__(self, json_path, fx=3):
        with open(json_path, 'r') as f:
            meme_cfg = json.loads(f.read())
        self.img = cv2.imread(meme_cfg['path'])
        self.points = np.array(meme_cfg['landmarks'])
#         self.points = np.array([(p[1], p[0]) for p in self.points])
        self.points, self.shape, self.face = self.extract_face()
        self.w, self.h = self.face.shape[:2]
        
    def extract_face(self, r=10):
        im_w, im_h = self.img.shape[:2]
        left, top = np.min(self.points, 0)
        right, bottom = np.max(self.points, 0)

        x, y = max(0, left - r), max(0, top - r)
        w, h = min(right + r, im_h) - x, min(bottom + r, im_w) - y

        return self.points - np.asarray([[x, y]]), (x, y, w, h), self.img[y:y+h, x:x+w]

    def swap_face(self, source_image, fa):
        try:
#         if True:
            src_points, src_shape, src_face = select_face_dl(source_image, fa)
#             src_points, src_shape, src_face = select_face_dl(source_image, detector, predictor)
            warped_src_face = warp_image_3d(src_face, src_points[:], self.points[:], (self.w, self.h))
            # Mask for blending
            mask = mask_from_points((self.w, self.h), self.points)
            mask_src = np.mean(warped_src_face, axis=2) > 0
            mask = np.asarray(mask*mask_src, dtype=np.uint8)
            # Correct color
            warped_src_face = apply_mask(warped_src_face, mask)
            dst_face_masked = apply_mask(self.face, mask)
            warped_src_face = correct_colours(dst_face_masked, warped_src_face, self.points)
            # Poisson Blending
            bounding_rect = cv2.boundingRect(mask)
            center = ((bounding_rect[0] + int(bounding_rect[2] / 2),
                       bounding_rect[1] + int(bounding_rect[3] / 2)))

            # output = cv2.seamlessClone(warped_src_face, self.face, mask, center, cv2.NORMAL_CLONE)

            mask = mask[..., None] // 255

            output = warped_src_face * mask + self.face * (1 - mask)
            # cv2.imshow('mask', output)
            # print(mask.max())

            x, y, w, h = self.shape
            result = self.img.copy()
            result[y:y+h, x:x+w] = output

            return result
        except Exception as e:
            print(e)
            return self.img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaceSwapApp')
    parser.add_argument('--src', required=True, help='Path for source image')
    parser.add_argument('--dst', required=True, help='Path for target image')
    parser.add_argument('--out', required=True, help='Path for storing output images')
    parser.add_argument('--warp_2d', default=False, action='store_true', help='2d or 3d warp')
    parser.add_argument('--correct_color', default=False, action='store_true', help='Correct color')
    parser.add_argument('--no_debug_window', default=False, action='store_true', help='Don\'t show debug window')
    args = parser.parse_args()
    

    # Read images
#     CNN_DETECTOR_PATH = '../FaceSwap/models/mmod_human_face_detector.dat '

#     detector = dlib.get_frontal_face_detector()
#     PREDICTOR_PATH = '../FaceSwap/models/shape_predictor_68_face_landmarks.dat'
#     predictor = dlib.shape_predictor(PREDICTOR_PATH)
    
    fa = fca.FaceAlignment(fca.LandmarksType._2D, device='cuda')   
    
    start_total = time()
    start = time()

    fx = 1

    # src_points, src_shape, src_face = select_face(src_img, detector, predictor)

    meme = Meme(args.dst, fx=fx)

    cap = cv2.VideoCapture('movie.mov')
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#     w, h = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter('output.mov', fourcc, 12.0, (meme.img.shape[1], meme.img.shape[0]))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = meme.swap_face(frame, fa)
        cv2.imwrite('res.jpg', result)
        writer.write(result)
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
