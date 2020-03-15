from os import listdir
from os.path import isfile, join
from random import shuffle, seed, randrange

import numpy as np
import cv2
from sklearn.feature_extraction.image import extract_patches_2d


class DataGenerator:

    def __init__(self, dataset_path, batch_size=128, min_dmos_difference=4.0, num_patches_for_image=8):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.min_dmos_difference = min_dmos_difference
        self.num_patches_for_image = num_patches_for_image

        seed(11)
        self.images = self._load_live2_db()


    def generate(self):
        def get_patches(image_path):
            img = cv2.imread(image_path)[...,::-1].astype(np.float32)
            img = self._local_contrast_normalization(img)
            patches = self._select_patches(
                self._extract_patches(img),
                self.num_patches_for_image
            )
            return [np.expand_dims(patch, axis=0) for patch in patches]

        while 1:
            y_true = []
            greater_dmos_batch, lower_dmos_batch = [], []
            while len(greater_dmos_batch) < self.batch_size:
                # We're looking for a pair of images so that the first one has the dmos value greater
                # than the second image at least by min_dmos_difference
                greater_dmos_img = self.images[randrange(len(self.images))]
                lower_dmos_img = self.images[randrange(len(self.images))]

                while greater_dmos_img['dmos'] < 5.0:
                    greater_dmos_img = self.images[randrange(len(self.images))]
                while (
                    greater_dmos_img == lower_dmos_img or
                    greater_dmos_img['dmos'] - lower_dmos_img['dmos'] < self.min_dmos_difference
                ):
                    lower_dmos_img = self.images[randrange(len(self.images))]

                greater_dmos_batch += get_patches(greater_dmos_img['path'])
                lower_dmos_batch   += get_patches(lower_dmos_img['path'])

            zipped_batches = list(zip(greater_dmos_batch, lower_dmos_batch))
            shuffle(zipped_batches)
            greater_dmos_batch, lower_dmos_batch = zip(*zipped_batches)

            greater_dmos_batch = np.moveaxis(np.array(greater_dmos_batch), 1, 3)
            lower_dmos_batch = np.moveaxis(np.array(lower_dmos_batch), 1, 3)    

            # the siamese architecture uses only the batch, not the second (y) value, thus setting to np.ones
            yield [greater_dmos_batch, lower_dmos_batch], np.ones((self.batch_size, 1))


    def _load_live2_db(self):
        distortion_types = ['fastfading', 'gblur', 'jp2k', 'jpeg', 'wn']
        images = []
        for distortion_type in distortion_types:
            distortion_path = join(self.dataset_path, distortion_type)
            with open(join(distortion_path, 'info.txt')) as info_file:
                lines = info_file.readlines()

            images += [{
                'path': join(distortion_path, img_description.split()[1]),
                'dmos': float(img_description.split()[2])
            } for img_description in lines if not img_description.isspace() and float(img_description.split()[2]) != 0.0]

        shuffle(images)
        return images


    def _local_contrast_normalization(self, img):
        kernel = (0, 0)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        float_gray = gray.astype(np.float32)

        blur = cv2.GaussianBlur(float_gray, kernel, sigmaX=3, sigmaY=3)
        num = float_gray - blur

        blur = cv2.GaussianBlur(num*num, kernel, sigmaX=3, sigmaY=3)
        den = cv2.pow(blur, 0.5) + np.finfo(float).eps

        gray = num / den

        return cv2.normalize(gray, dst=gray, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)


    def _extract_patches(self, image):
        return extract_patches_2d(
            image=image,
            patch_size=(64, 64),
            max_patches=300,
            random_state=3,
        )


    def _select_patches(self, patches, n=8):
        # Selects n patches from a patch sequence sorted by patches standart deviations
        half_n = int(n / 2)
        mean_values = [np.std(p) for p in patches]
        sorted_sequence = [p for _, p in sorted(zip(mean_values, patches), key=lambda x: x[0])]
        return sorted_sequence[int(len(patches) / 2) - half_n : int(len(patches) / 2) + half_n]
