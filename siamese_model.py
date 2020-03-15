import os
from math import ceil

from keras.layers import Flatten, Dense, Input, Lambda, Conv2D, MaxPooling2D, Dropout
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam


class SiameseCNN:

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.input_shape = (64, 64, 1)

        input = Input(shape=self.input_shape, name='image_input')

        # CNN layers as described in the original paper
        cnn_model = Sequential()
        cnn_model.add(self._cnn_layer())
        cnn_model.add(self._cnn_layer())
        cnn_model.add(self._max_pool_layer())
        cnn_model.add(self._cnn_layer())
        cnn_model.add(self._cnn_layer())
        cnn_model.add(self._max_pool_layer())
        cnn_model.add(self._cnn_layer())
        cnn_model.add(self._cnn_layer())
        cnn_model.add(self._cnn_layer())
        cnn_model.add(self._max_pool_layer())
        cnn_model.add(self._cnn_layer())
        cnn_model.add(self._cnn_layer())
        cnn_model.add(self._cnn_layer())
        cnn_model.add(self._max_pool_layer())
        cnn_model.add(self._cnn_layer())
        cnn_model.add(self._cnn_layer())
        cnn_model.add(self._cnn_layer())

        cnn_model.add(Flatten(name='flatten'))
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
            optimizer=Adam(lr=0.0004),
            loss=self._contrastive_loss, 
            metrics=[self._accuracy]
        )

        self.model.fit_generator(
            generator=data_generator.generate(),
            steps_per_epoch = ceil(70000 / self.batch_size),
            epochs=1
        )


    def _l2_norm(self, x):
        return K.l2_normalize(x, axis=-1)


    def _contrastive_loss(self, positive_pair, y_pred):
        max_IQA = 100.0
        # if positive_pair[:, 0] == 1:
        #     return (2 / max_IQA) * y_pred**2
        # else:
        #     return 2 * max_IQA * K.exp(-(2.77 * y_pred) / max_IQA)
        return (2 / max_IQA) * y_pred**2


    def _euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))


    def _eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)


    def _accuracy(self, y_true, y_pred):
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, 'float32')))
        # return K.mean(K.equal(y_true, float(y_pred < 0.5)))


    def _cnn_layer(self):
        return Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=self.input_shape)


    def _max_pool_layer(self):
        return MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
