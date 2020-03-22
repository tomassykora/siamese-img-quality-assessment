from os import listdir
from os.path import isfile, join
from random import shuffle, seed, randrange

import numpy as np
import cv2
from sklearn.feature_extraction.image import extract_patches_2d

from utils import get_preprocessed_patches


class DataGenerator:

    def __init__(self, dataset_path, batch_size=128, min_dmos_difference=4.0, num_patches_for_image=8, class_margin=0.3):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.min_dmos_difference = min_dmos_difference
        self.num_patches_for_image = num_patches_for_image
        self.class_margin = class_margin

        seed(11)
        self.images = self._load_live2_db()


    def generate(self):
        while 1:
            greater_dmos_batch, lower_dmos_batch, y_true = [], [], []
            while len(greater_dmos_batch) < self.batch_size:
                greater_dmos_img = self.images[randrange(len(self.images))]
                lower_dmos_img = self.images[randrange(len(self.images))]
                while (
                    greater_dmos_img == lower_dmos_img or
                    abs(greater_dmos_img['dmos'] - lower_dmos_img['dmos']) < self.min_dmos_difference
                ):
                    lower_dmos_img = self.images[randrange(len(self.images))]

                greater_dmos_batch += get_preprocessed_patches(greater_dmos_img['path'])
                lower_dmos_batch   += get_preprocessed_patches(lower_dmos_img['path'])
                y_true += [np.array([1.0, 0.0]) if greater_dmos_img['dmos'] > lower_dmos_img['dmos'] else np.array([0.0, 1.0])
                            for i in range(self.num_patches_for_image)]

            zipped_batches = list(zip(greater_dmos_batch, lower_dmos_batch, y_true))
            shuffle(zipped_batches)
            greater_dmos_batch, lower_dmos_batch, y_true = zip(*zipped_batches)

            # y_true += [1.0 if greater_dmos_img['dmos'] > lower_dmos_img['dmos'] else 0.0 for i in range(self.num_patches_for_image)]
            # for batch in zip(greater_dmos_batch, lower_dmos_batch):
            #     g, l = batch
            #     print(g.shape, l.shape)
            #     y_true.append(1.0 if g['dmos'] > l['dmos'] else 0.0)

            greater_dmos_batch = np.moveaxis(np.array(greater_dmos_batch), 1, 3)
            lower_dmos_batch = np.moveaxis(np.array(lower_dmos_batch), 1, 3)

            # the siamese architecture uses only the batch, not the second (y) value, thus setting to np.ones
            # yield [greater_dmos_batch, lower_dmos_batch], np.ones((self.batch_size, 1))
            # yield [greater_dmos_batch, lower_dmos_batch], np.array(y_true).reshape(len(np.array(y_true)), 2)
            yield [greater_dmos_batch, lower_dmos_batch], np.array(y_true)


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
