import json
import argparse

import cv2
import numpy as np

import face_alignment

from face_swap import warp_image_3d, mask_from_points, apply_mask, correct_colours


class Meme():
    def __init__(self, json_path, device='cuda'):
        '''
        Parameters:
        -----------
            json_path: str
                path to the json with the meme's meta information
            device: {'cuda', 'cpu'}
                the device to run inference on
        '''

        with open(json_path, 'r') as f:
            meme_cfg = json.loads(f.read())
        self.img = cv2.imread(meme_cfg['path'])
        self.points = np.array(meme_cfg['landmarks'])
        self.points, self.shape, self.face = self._extract_face(self.img, self.points)
        self.w, self.h = self.face.shape[:2]
        
        self.landmarks_extractor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device)
        
    def _get_landmarks(self, img):
        points = self.landmarks_extractor.get_landmarks(img)
        # if no face is found
        if points is None:
            return None, None, None

        return points[0]
        
    def _extract_face(self, img, points, r=10):
        '''Get face from the `img`
        
        Similar to `select_face_dl` except the dl part
        
        Parameters:
        -----------
            img: np.ndarray (h, w)
                source image
            points: sequence (68, 2)
                face's landmarks
            r: int
                face's bounding box is expanded by the value of `r`
        
        Returns:
        --------
            points: np.ndarray (68, 2)
                face landmarks shifted to the origin
            bounging box: tuple
                left, top, width, height of the face's bounding box
            face: np.ndarray (h, w)
                extracted face
        '''

        im_w, im_h = img.shape[:2]
        left, top = np.min(points, 0)
        right, bottom = np.max(points, 0)
        
        x, y = int(max(0, left - r)), int(max(0, top - r))
        w, h = int(min(right + r, im_h) - x), int(min(bottom + r, im_w) - y)

        return points - np.asarray([[x, y]]), (x, y, w, h), img[y:y+h, x:x+w]

    def swap_face(self, source_image):
        try:
            src_points = self._get_landmarks(source_image)
            src_points, src_shape, src_face = self._extract_face(source_image, src_points)
            warped_src_face = warp_image_3d(src_face, src_points[:], self.points[:], (self.w, self.h))
            # Mask for blending
            mask = mask_from_points((self.w, self.h), self.points)
            mask_src = np.mean(warped_src_face, axis=2) > 0
            mask = np.asarray(mask*mask_src, dtype=np.uint8)

            # Correct color
            warped_src_face = apply_mask(warped_src_face, mask)
            dst_face_masked = apply_mask(self.face, mask)
            warped_src_face = correct_colours(dst_face_masked, warped_src_face, self.points)

            # Poisson Blending. Might be used in future
            # bounding_rect = cv2.boundingRect(mask)
            # center = ((bounding_rect[0] + int(bounding_rect[2] / 2),
            #            bounding_rect[1] + int(bounding_rect[3] / 2)))
            # output = cv2.seamlessClone(warped_src_face, self.face, mask, center, cv2.NORMAL_CLONE)

            # clone face
            mask = mask[..., None] // 255
            output = warped_src_face * mask + self.face * (1 - mask)
            x, y, w, h = self.shape
            result = self.img.copy()
            result[y:y+h, x:x+w] = output

            return result
        except Exception as e:
            print(e)
            return self.img

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of face swapping')
    parser.add_argument('--meme', required=True, help='Path for meme data', default='memes/trololo_face.json')
    parser.add_argument('--src', required=True, help='Source image', default='open_mouth.jpg')
    parser.add_argument('--out', required=True, help='Result image', default='kek.jpg')
    parser.add_argument('--device', required=True, help='Device to run inference on. Can be one of [gpu, cpu].', default='cpu')
    args = parser.parse_args()
    
    print('--> Initializing meme. This can take a while on the first run (due to the neural networks\' weights downloading)')
    meme = Meme(args.meme, 'cuda' if args.device == 'gpu' else 'cpu')
    
    print('--> Loading source image')
    source_image = cv2.imread(args.src)
    if source_image is None:
        print("Image is not loaded. Aborting.")
        sys.exit(1)
    
    print('--> Swapping faces')
    result = meme.swap_face(source_image)
    
    print('--> Writing result')
    cv2.imwrite(args.out, result)
    
    print('--> All done. Bye!')