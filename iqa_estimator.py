from os import path
import numpy as np
from keras import backend as K

from scipy.stats import spearmanr

from utils import get_preprocessed_patches


class IQAEstimator:

    def __init__(self, model, test_images, num_patches_for_image=8):
        self.tid2013_images = []
        self.num_patches_for_image = num_patches_for_image
        self.test_images = test_images
        self.model = model
        self._prepare_test_patches()


    def full_test_set_eval(self):
        for test_img in self.test_images:
            test_img['estimated_iqa'] = 0
            for img_to_compare in self.test_images:
                if test_img != img_to_compare:
                    test_img['estimated_iqa'] += self._compare_image_patches(
                        test_img['patches'],
                        img_to_compare['patches']
                    )
        
        rho, p_val = spearmanr(
            [img['iqa'] for img in self.test_images],
            [img['estimated_iqa'] for img in self.test_images]
        )

        print(f'Spearman rank-order correlation coefficients are: rho {rho}, p-val: {p_val}')


    def _prepare_test_patches(self):
        for test_img in self.test_images:
            test_img['patches'] = get_preprocessed_patches(test_img['path'], self.num_patches_for_image)


    def _compare_image_patches(self, patches, test_patches):
        patches_batch, test_patches_batch = [], []
        for p in patches:
            for p_test in test_patches:
                patches_batch.append(p)
                test_patches_batch.append(p_test)

        patches_batch = np.moveaxis(np.array(patches_batch), 1, 3)
        test_patches_batch = np.moveaxis(np.array(test_patches_batch), 1, 3)

        pred = self.model.predict([patches_batch, test_patches_batch])

        # Make average over all image patches
        # higher_iqa = sum([p[0] for p in pred]) / len(pred)
        # lower_iqa = sum([p[1] for p in pred]) / len(pred)
        # return [higher_iqa, lower_iqa]

        # The original paper sums ones (one of tested image has higher iqa, zero elsewhere)
        return sum([int(p[0] > p[1]) for p in pred])
