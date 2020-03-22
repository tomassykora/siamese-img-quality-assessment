import numpy as np
from keras import backend as K

from utils import get_preprocessed_patches


class IQAEstimator:

    def __init__(self, model, num_patches_for_image=8):
        self.num_patches_for_image = num_patches_for_image
        self.test_images = [
            {'path': 'tid2013/tid2013/distorted_images/i02_23_1.bmp', 'iqa_val': 6.77778, 'patches': []},
            {'path': 'tid2013/tid2013/distorted_images/I01_01_1.bmp', 'iqa_val': 5.51429, 'patches': []},
            {'path': 'tid2013/tid2013/distorted_images/i01_17_3.bmp', 'iqa_val': 4.08108, 'patches': []},
            {'path': 'tid2013/tid2013/distorted_images/i01_08_4.bmp', 'iqa_val': 3.00000, 'patches': []},
            {'path': 'tid2013/tid2013/distorted_images/i01_10_5.bmp', 'iqa_val': 2.08108, 'patches': []},
            {'path': 'tid2013/tid2013/distorted_images/i17_11_5.bmp', 'iqa_val': 1.02500, 'patches': []},
            {'path': 'tid2013/tid2013/distorted_images/i09_24_5.bmp', 'iqa_val': 0.24242, 'patches': []}
        ]
        self.model = model
        self._prepare_test_patches()


    def evaluate_image(self, img_path):
        patches = get_preprocessed_patches(img_path, self.num_patches_for_image)
        results = []
        for test_img in self.test_images:
            results.append(self._compare_image_patches(patches, test_img['patches']))

        print(results)


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

        # Make average over all imae patches
        higher_iqa = sum([p[0] for p in pred]) / len(pred)
        lower_iqa = sum([p[1] for p in pred]) / len(pred)

        return [higher_iqa, lower_iqa]
