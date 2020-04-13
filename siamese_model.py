import os
from math import ceil
import numpy as np

from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers.core import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam

from utils import get_preprocessed_patches


class SiameseCNN:

    def __init__(self, batch_size, data_generator, orig_arch=True, model_filename='model.h5'):
        self.batch_size = batch_size
        self.input_shape = (64, 64, 1)
        self.data_generator = data_generator
        self.orig_arch = orig_arch
        self.model_filename = model_filename

        input = Input(shape=self.input_shape, name='image_input')

        # CNN layers as described in the original paper
        cnn_model = Sequential()
        cnn_model.add(self._cnn_layer(filters=16, input_shape=self.input_shape))
        cnn_model.add(self._cnn_layer(filters=16))
        cnn_model.add(self._max_pool_layer(strides=1))
        cnn_model.add(self._cnn_layer(filters=32))
        cnn_model.add(self._cnn_layer(filters=32))
        cnn_model.add(self._max_pool_layer(strides=1))
        cnn_model.add(self._cnn_layer(filters=48))
        cnn_model.add(self._cnn_layer(filters=48))
        cnn_model.add(self._cnn_layer(filters=48))
        cnn_model.add(self._max_pool_layer(strides=2))
        cnn_model.add(self._cnn_layer(filters=64))
        cnn_model.add(self._cnn_layer(filters=64))
        cnn_model.add(self._cnn_layer(filters=64))
        cnn_model.add(self._max_pool_layer(strides=2))
        cnn_model.add(self._cnn_layer(filters=80))
        cnn_model.add(self._cnn_layer(filters=80))
        cnn_model.add(self._cnn_layer(filters=80))
        cnn_model.add(self._max_pool_layer(strides=2))
        cnn_model.add(Flatten(name='flatten'))
        

        if self.orig_arch:
            self.base_network = cnn_model
            self.base_network.summary()

            input_a = Input(shape=self.input_shape)
            input_b = Input(shape=self.input_shape)

            out = Concatenate()([
                self.base_network(input_a),
                self.base_network(input_b)
            ])

            out = Dense(512, activation='relu', name='fc1')(out)
            # out = Dropout(0.5)(out)
            out = Dense(512, activation='relu', name='fc2')(out)
            # out = Dropout(0.5)(out)
            out = Dense(2, name='fc3')(out)
            out = Dense(2, activation='softmax', name='fc4')(out)

            self.model = Model([input_a, input_b], out)
        else:
            cnn_model.add(Dense(512, activation='relu', name='fc1'))
            cnn_model.add(Dropout(0.5))
            cnn_model.add(Dense(512, name='embeded'))
            cnn_model.add(Lambda(self._l2_norm, output_shape=[512]))

            self.base_network = cnn_model
            self.base_network.summary()

            input_a = Input(shape=self.input_shape)
            input_b = Input(shape=self.input_shape)

            processed_a = self.base_network(input_a)
            processed_b = self.base_network(input_b)

            distance = Lambda(
                self._euclidean_distance,
                output_shape=self._eucl_dist_output_shape
            )([processed_a, processed_b])

            self.model = Model([input_a, input_b], distance)

        self.model.summary()


    def train(self, data_generator):
        self.model.compile(
            optimizer=Adam(lr=0.00004),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model.fit_generator(
            generator=self.data_generator.generate(),
            steps_per_epoch = ceil(700000 / self.batch_size),
            epochs=2
        )

        # serialize weights to HDF5
        self.model.save(self.model_filename)


    def _l2_norm(self, x):
        return K.l2_normalize(x, axis=-1)


    def _contrastive_loss(self, positive_pair, y_pred):
        max_IQA = 100.0
        # if positive_pair[:, 0] == 1:
        #     return (2 / max_IQA) * y_pred**2
        # else:
        #     return 2 * max_IQA * K.exp(-(2.77 * y_pred) / max_IQA)

        return (1.0 - positive_pair) * (2 * max_IQA * K.exp(-(2.77 * y_pred) / max_IQA)) + positive_pair * ((2 / max_IQA) * y_pred**2)
        # return (2 / max_IQA) * y_pred**2


    def _euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))


    def _eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)


    def _accuracy(self, y_true, y_pred):
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, 'float32')))


    def _cnn_layer(self, filters, input_shape=None):
        if input_shape:
            return Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape)
        else:
            return Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')


    def _max_pool_layer(self, strides):
        return MaxPooling2D(pool_size=(2, 2), strides=strides, padding='valid')
