# Image quality assessment based on https://ieeexplore.ieee.org/abstract/document/8771231
# Tom Sykora (tms.sykora@gmail.com), 2020

from siamese_model import SiameseCNN
from data_generator import DataGenerator


def main():
    model = SiameseCNN()
    data_generator = DataGenerator('path/to/dataset')

    model.train(data_generator)


if __name__ == '__main__':
    main()
