from os import listdir
from os.path import isfile, join
from random import shuffle, seed, randrange, uniform
from multiprocessing import Pool

import numpy as np
import cv2
from sklearn.feature_extraction.image import extract_patches_2d

from utils import get_preprocessed_patches


class DataGenerator:

    def __init__(self, dataset_path, batch_size=128, min_iqa_difference=4.0, num_patches_for_image=8):
        assert batch_size % num_patches_for_image == 0

        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.min_iqa_difference = min_iqa_difference
        self.num_patches_for_image = num_patches_for_image

        seed(11)
        self.images, self.test_images = self._load_live2_db()
        print(f'LOG: number of train images: {len(self.images)}')

        self.prepare_image_pairs()
        print(f'LOG: number of patch pairs: {len(self.pairs)}')


    def generate(self):
        images_per_batch = int(self.batch_size / self.num_patches_for_image)
        while 1:
            for i in range(0, len(self.pairs), images_per_batch):
                batch_pairs = self.pairs[i:i + images_per_batch]

                y_true = []
                for pair in batch_pairs:
                    y_true += [np.array([np.float32(1.0), np.float32(0.0)]) if pair[0]['iqa'] > pair[1]['iqa'] else np.array([np.float32(0.0), np.float32(1.0)])
                                for i in range(self.num_patches_for_image)]

                iqa_batch_a = [get_preprocessed_patches(pair[0]['path']) for pair in batch_pairs]
                iqa_batch_a = [patch for img_patches in iqa_batch_a for patch in img_patches]

                iqa_batch_b = [get_preprocessed_patches(pair[1]['path']) for pair in batch_pairs]
                iqa_batch_b = [patch for img_patches in iqa_batch_b for patch in img_patches]

                zipped_batches = list(zip(iqa_batch_a, iqa_batch_b, y_true))
                shuffle(zipped_batches)
                iqa_batch_a, iqa_batch_b, y_true = zip(*zipped_batches)

                iqa_batch_a = np.moveaxis(np.stack(iqa_batch_a), 1, 3)
                iqa_batch_b = np.moveaxis(np.stack(iqa_batch_b), 1, 3)

                yield [iqa_batch_a, iqa_batch_b], np.stack(y_true)


    def prepare_image_pairs(self):
        self.pairs = []
        one_zero = 0
        zero_one = 0
        for img_a in self.images:
            for img_b in self.images:
                if (
                    img_a == img_b or
                    abs(img_a['iqa'] - img_b['iqa']) < self.min_iqa_difference
                ):
                    continue
                if img_a['iqa'] > img_b['iqa']:
                    one_zero += 1
                else:
                    zero_one +=1

                self.pairs.append((img_a, img_b, i, j))
        shuffle(self.pairs)

        print('DATASET BALANCE: ', one_zero, zero_one)


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

        # Splitting in 80/20 ratio
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
