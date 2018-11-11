#! /usr/bin/env python
import os
import cv2
import dlib
import argparse
import json

import numpy as np

from face_detection import face_detection
from face_points_detection import face_points_detection
from face_swap import warp_image_2d, warp_image_3d, mask_from_points, apply_mask, correct_colours, transformation_from_points
from time import time


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
    def __init__(self, json_path, detector, predictor, fx=3):
        with open(json_path, 'r') as f:
            meme_cfg = json.loads(f.read())
        self.img = cv2.imread(meme_cfg['path'])
        self.points = np.array(meme_cfg['landmarks'])
        self.points, self.shape, self.face = self.extract_face()
        self.w, self.h = self.face.shape[:2]
        
    def extract_face(self, r=10):
        im_w, im_h = self.img.shape[:2]
        left, top = np.min(self.points, 0)
        right, bottom = np.max(self.points, 0)

        x, y = max(0, left - r), max(0, top - r)
        w, h = min(right + r, im_h) - x, min(bottom + r, im_w) - y

        return self.points - np.asarray([[x, y]]), (x, y, w, h), self.img[y:y+h, x:x+w]

    def swap_face(self, source_image):
        try:
            src_points, src_shape, src_face = select_face(source_image, detector, predictor)
            start = time()
            warped_src_face = warp_image_3d(src_face, src_points[:], self.points[:], (self.w, self.h))
            # Mask for blending
            mask = mask_from_points((self.w, self.h), self.points)
            mask_src = np.mean(warped_src_face, axis=2) > 0
            mask = np.asarray(mask*mask_src, dtype=np.uint8)
            # dilate mask a little
            # kernel = np.array(
            #     [[0, 0, 0, 0, 0],
            #      [0, 0, 0, 0, 0],
            #      [0, 0, 1, 0, 0],
            #      [0, 0, 1, 0, 0],
            #      [0, 0, 1, 0, 0]]
            # ).astype(np.uint8)
            # mask = cv2.dilate(mask, kernel, iterations=2)

            # cv2.imshow('dilated', keke_mask-mask)
            # Correct color
            warped_src_face = apply_mask(warped_src_face, mask)
            dst_face_masked = apply_mask(self.face, mask)
            warped_src_face = correct_colours(dst_face_masked, warped_src_face, self.points)
            # cv2.imshow('warped', warped_src_face)
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
        except:
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
    CNN_DETECTOR_PATH = '../FaceSwap/models/mmod_human_face_detector.dat '

    detector = dlib.get_frontal_face_detector()
    PREDICTOR_PATH = '../FaceSwap/models/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    start_total = time()
    start = time()

    fx = 1

    # src_points, src_shape, src_face = select_face(src_img, detector, predictor)

    meme = Meme(args.dst, detector, predictor, fx=fx)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        src_img = cv2.resize(cv2.GaussianBlur(frame, (3, 3), 0), (0, 0), fx=1/fx, fy=1/fx)
        result = meme.swap_face(src_img)
        cv2.imshow("result", result)
        q = cv2.waitKey(1) & 0xff
        if q == ord('q'):
            break

    # dst_img = cv2.imread(args.dst)
    # dst_img = cv2.resize(cv2.GaussianBlur(cv2.imread(args.dst), (3, 3), 0), (0, 0), fx=1/fx, fy=1/fx)

    # print('reading', time()-start)
    # start = time()

    # Select src face
    # src_points, src_shape, src_face = select_face(src_img, detector, predictor)
    # Select dst face
    # dst_points, dst_shape, dst_face = select_face(dst_img, detector, predictor)

#     def draw_keys(img, keys):
#         img_with_keys = img.copy()
#         for k in keys:
#             cv2.circle(img_with_keys, tuple([k[0], k[1]]), 1, (0, 0, 255))
#         return img_with_keys
#
# #     print(dst_points)
# #     cv2.imshow("src_face", draw_keys(src_face, src_points))
# #     cv2.imshow("dst_face", draw_keys(dst_face, dst_points))
#
#
# #     bbox = [
# #         int((src_face.top()-2)),
# #         int((src_face.left()-2)),
# #         int((src_face.bottom()+2)),
# #         int((src_face.right()+2)),
# #     ]
#     cv2.imshow('face', dst_face)
#     print('detecting landmarks', time()-start)
#     start = time()
#
#     w, h = dst_face.shape[:2]
#
#     # Warp Image
#     if not args.warp_2d:
#         # 3d warp
#         # warped_src_face = warp_image_3d(src_face, src_points[:48], dst_points[:48], (w, h))
#         warped_src_face = warp_image_3d(src_face, src_points[:], dst_points[:], (w, h))
#     else:
#         # 2d warp
#         src_mask = mask_from_points(src_face.shape[:2], src_points)
#         src_face = apply_mask(src_face, src_mask)
#         # Correct Color for 2d warp
#         if args.correct_color:
#             warped_dst_img = warp_image_3d(dst_face, dst_points[:48], src_points[:48], src_face.shape[:2])
#             src_face = correct_colours(warped_dst_img, src_face, src_points)
#         # Warp
#         warped_src_face = warp_image_2d(src_face, transformation_from_points(dst_points, src_points), (w, h, 3))
#
#     print('warping', time()-start)
#     start = time()
#     # Mask for blending
#     mask = mask_from_points((w, h), dst_points)
#     mask_src = np.mean(warped_src_face, axis=2) > 0
#     mask = np.asarray(mask*mask_src, dtype=np.uint8)
#
#     print('masking', time()-start)
#     start = time()
#     # Correct color
#     if not args.warp_2d and args.correct_color:
#         warped_src_face = apply_mask(warped_src_face, mask)
#         dst_face_masked = apply_mask(dst_face, mask)
#         warped_src_face = correct_colours(dst_face_masked, warped_src_face, dst_points)
#     print('correcting', time()-start)
#     start = time()
#     # Shrink the mask
#     kernel = np.ones((5, 5), np.uint8)
#     # mask = cv2.erode(mask, kernel, iterations=1)
#     print('shrinking', time()-start)
#     start = time()
#     # Poisson Blending
#     r = cv2.boundingRect(mask)
#     center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
#     output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)
# #     output=
#     print('blending', time()-start)
#     start = time()
#     print('total', time()-start_total)
#     x, y, w, h = dst_shape
#     dst_img_cp = dst_img.copy()
#     dst_img_cp[y:y+h, x:x+w] = output
#     output = dst_img_cp

    # dir_path = os.path.dirname(args.out)
    # if not os.path.isdir(dir_path):
    # os.makedirs(dir_path)

    # cv2.imwrite(args.out, result)

    # For debug
    # if not args.no_debug_window:
    # cv2.imshow("src", src_img)
    # cv2.imshow("From", meme.img)


cv2.destroyAllWindows()
