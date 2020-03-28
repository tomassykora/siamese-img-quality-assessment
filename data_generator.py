from os import listdir
from os.path import isfile, join
from random import shuffle, seed, randrange, uniform

import numpy as np
import cv2
from sklearn.feature_extraction.image import extract_patches_2d

from utils import get_preprocessed_patches


class DataGenerator:

    def __init__(self, dataset_path, batch_size=128, min_iqa_difference=4.0, num_patches_for_image=8, class_margin=0.3):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.min_iqa_difference = min_iqa_difference
        self.num_patches_for_image = num_patches_for_image
        self.class_margin = class_margin

        seed(11)
        self.images, self.test_images = self._load_live2_db()


    def generate(self):
        while 1:
            iqa_batch_a, iqa_batch_b, y_true = [], [], []
            while len(iqa_batch_a) < self.batch_size:
                greater_iqa_img = self.images[randrange(len(self.images))]
                lower_iqa_img = self.images[randrange(len(self.images))]
                while (
                    greater_iqa_img == lower_iqa_img or
                    abs(greater_iqa_img['iqa'] - lower_iqa_img['iqa']) < self.min_iqa_difference
                ):
                    lower_iqa_img = self.images[randrange(len(self.images))]

                iqa_batch_a += get_preprocessed_patches(greater_iqa_img['path'])
                iqa_batch_b   += get_preprocessed_patches(lower_iqa_img['path'])
                y_true += [np.array([1.0, 0.0]) if greater_iqa_img['iqa'] > lower_iqa_img['iqa'] else np.array([0.0, 1.0])
                            for i in range(self.num_patches_for_image)]

            zipped_batches = list(zip(iqa_batch_a, iqa_batch_b, y_true))
            shuffle(zipped_batches)
            iqa_batch_a, iqa_batch_b, y_true = zip(*zipped_batches)

            # y_true += [1.0 if greater_iqa_img['iqa'] > lower_iqa_img['iqa'] else 0.0 for i in range(self.num_patches_for_image)]
            # for batch in zip(iqa_batch_a, iqa_batch_b):
            #     g, l = batch
            #     print(g.shape, l.shape)
            #     y_true.append(1.0 if g['iqa'] > l['iqa'] else 0.0)

            iqa_batch_a = np.moveaxis(np.array(iqa_batch_a), 1, 3)
            iqa_batch_b = np.moveaxis(np.array(iqa_batch_b), 1, 3)

            # the siamese architecture uses only the batch, not the second (y) value, thus setting to np.ones
            # yield [iqa_batch_a, iqa_batch_b], np.ones((self.batch_size, 1))
            # yield [iqa_batch_a, iqa_batch_b], np.array(y_true).reshape(len(np.array(y_true)), 2)
            yield [iqa_batch_a, iqa_batch_b], np.array(y_true)


    def _load_live2_db(self):
        distortion_types = ['fastfading', 'gblur', 'jp2k', 'jpeg', 'wn']
        images = []
        for distortion_type in distortion_types:
            distortion_path = join(self.dataset_path, distortion_type)
            with open(join(distortion_path, 'info.txt')) as info_file:
                lines = info_file.readlines()

            images += [{
                'path': join(distortion_path, img_description.split()[1]),
                'iqa': float(img_description.split()[2])
            } for img_description in lines if not img_description.isspace() and float(img_description.split()[2]) != 0.0]
        shuffle(images)

        # Splitting in 80/20% ratio
        train_images = images[:int(0.8 * len(images))]
        test_images  = images[int(0.8 * len(images)):]

        return train_images, test_images


    def _load_tid2013_db(self):
        with open(path.join(self.dataset_path, 'mos_with_names.txt')) as f:
            images_info = f.readlines()

        images = []
        for inf in images_info:
            images.append({
                'path': path.join(self.dataset_path, 'distorted_images', inf.split()[1]),
                'iqa': float(inf.split()[0])
            })
        shuffle(images)

        # Splitting in 80/20% ratio
        train_images = images[:int(0.8 * len(images))]
        test_images  = images[int(0.8 * len(images)):]

        return train_images, test_images
