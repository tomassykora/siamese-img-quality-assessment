#!/usr/bin/env python3.6

# Image quality assessment based on https://ieeexplore.ieee.org/abstract/document/8771231
# Tom Sykora (tms.sykora@gmail.com), 2020

from keras.models import load_model

from siamese_model import SiameseCNN
from data_generator import DataGenerator
from iqa_estimator import IQAEstimator


def main():
    batch_size = 64

    data_generator = DataGenerator(
        batch_size=batch_size,
        dataset_path='/Users/tomassykora/Projects/school/siamese-img-quality-assessment/live2'
    )

    model_filename = 'model.h5'
    model = SiameseCNN(
        batch_size,
        data_generator,
        model_filename=model_filename
    )
    model.train(data_generator)

    estimator = IQAEstimator(
        model=load_model(model_filename),
        test_images=data_generator.test_images
    )
    estimator.full_test_set_eval()


if __name__ == '__main__':
    main()
