from os import listdir
from os.path import isfile, join
from random import shuffle, seed, randrange

import numpy as np
import cv2
from keras.preprocessing import image
from sklearn.feature_extraction.image import extract_patches_2d


class DataGenerator:

    def __init__(self, dataset_path, batch_size=128, min_dmos_difference=4.0):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.min_dmos_difference = min_dmos_difference

        seed(7)
        self._load_live2_db()


    def generate(self):
        def get_patches(image_path):
            img = cv2.imread(image_path)[...,::-1].astype(np.float32)
            img = self._local_contrast_normalization(img)
            patches = self._select_patches(self._extract_patches(img))
            return [np.expand_dims(patch, axis=0) for patch in patches]

        while 1:
            greater_dmos_batch, lower_dmos_batch = [], []
            while len(greater_dmos_batch) < self.batch_size:
                # We're looking for a pair of images so that the first one has the dmos value greater
                # than the second image at least by min_dmos_difference
                img1 = self.images[randrange(len(self.images))]
                img2 = self.images[randrange(len(self.images))]
                while (
                    img1 == img2 and
                    img1['dmos'] - img2['dmos'] >= self.min_dmos_difference
                ):
                    img2 = self.images[randrange(len(self.images))]

                greater_dmos_batch += get_patches(img1['path'])
                lower_dmos_batch   += get_patches(img2['path'])

            # the siamese architecture uses only the batch, not the second (y) value, thus setting to np.ones
            yield [np.array(greater_dmos_batch), np.array(lower_dmos_batch)], np.ones((self.batch_size, 1, 1))


    def _load_live2_db(self):
        distortion_types = ['fastfading', 'gblur', 'jp2k', 'jpeg', 'wn']
        self.images = []
        for distortion_type in distortion_types:
            with open('info.txt') as info_file:
                lines = info_file.readlines()

            self.images += [{
                'path': join(self.dataset_path, distortion_type, img_description[1]),
                'dmos': img_description[2]
            } for l in lines]

        shuffle(self.images)


    def _local_contrast_normalization(self, img):
        k = l = 3  # normalization kernel size

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        float_gray = gray.astype(np.float32) / 255.0

        blur = cv2.GaussianBlur(float_gray, (k, l), sigmaX=2, sigmaY=2)
        num = float_gray - blur

        blur = cv2.GaussianBlur(num*num, (k, l), sigmaX=20, sigmaY=20)
        den = cv2.pow(blur, 0.5)

        gray = num / den

        return cv2.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX) * 255


    def _extract_patches(self, image):
        return extract_patches_2d(
            image=image,
            patch_size=(64, 64),
            max_patches=300,
            random_state=3,
        )


    def _select_patches(self, patches, n=8):
        # Selects n patches from a patch sequence sorted by patches standart deviations
        mean_values = [np.std(p) for p in patches]
        sorted_sequence = [p for _, p in sorted(zip(mean_values, patches))]
        return sorted_sequence[int(len(patches) / 2) - n : int(int(len(patches) / 2) + n)]
