#!/usr/bin/env python3.6

# Image quality assessment based on https://ieeexplore.ieee.org/abstract/document/8771231
# Tom Sykora (tms.sykora@gmail.com), 2020

from siamese_model import SiameseCNN
from data_generator import DataGenerator


def main():
    batch_size = 128

    model = SiameseCNN(batch_size)
    data_generator = DataGenerator(
        batch_size=batch_size,
        dataset_path='/Users/tomassykora/Projects/school/siamese-img-quality-assessment/live2'
    )

    model.train(data_generator)


if __name__ == '__main__':
    main()
