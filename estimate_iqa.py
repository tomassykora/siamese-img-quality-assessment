from keras.models import load_model
from iqa_estimator import IQAEstimator


def main():
    estimator = IQAEstimator(load_model('model.h5'))

    estimator.evaluate_image('tid2013/tid2013/distorted_images/i01_07_4.bmp')  # iqa value: 3.22222


if __name__ == '__main__':
    main()
