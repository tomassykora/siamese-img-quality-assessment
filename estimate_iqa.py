#!/usr/bin/env python3.6

from keras.models import load_model

from data_generator import DataGenerator
from iqa_estimator import IQAEstimator


def main():
    data_generator = DataGenerator(
        dataset_path='/Users/tomassykora/Projects/school/siamese-img-quality-assessment/live2'
    )

    estimator = IQAEstimator(
        model=load_model('model.h5'),
        test_images=data_generator.test_images
    )
    estimator.full_test_set_eval()


if __name__ == '__main__':
    main()
