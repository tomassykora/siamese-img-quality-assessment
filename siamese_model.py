from keras.layers import Flatten, Dense, Input, Lambda, Conv2D, MaxPooling2D, Dropout
from keras.models import Model, Sequential
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adam


class SiameseCNN:

    def __init__(self):
        self.input_shape = (64, 64, 3)

        input = Input(shape=input_shape, name='image_input')

        # cnn_model = InceptionV3(weights='imagenet', include_top=False)
        # cnn_model.trainable = False

        # CNN layers as described in the original paper
        cnn_model = Sequential()
        cnn_model = self._cnn_layer(cnn_model)
        cnn_model = self._cnn_layer(cnn_model)
        cnn_model = self._max_pool_layer(cnn_model)
        cnn_model = self._cnn_layer(cnn_model)
        cnn_model = self._cnn_layer(cnn_model)
        cnn_model = self._max_pool_layer(cnn_model)
        cnn_model = self._cnn_layer(cnn_model)
        cnn_model = self._cnn_layer(cnn_model)
        cnn_model = self._cnn_layer(cnn_model)
        cnn_model = self._max_pool_layer(cnn_model)
        cnn_model = self._cnn_layer(cnn_model)
        cnn_model = self._cnn_layer(cnn_model)
        cnn_model = self._cnn_layer(cnn_model)
        cnn_model = self._max_pool_layer(cnn_model)
        cnn_model = self._cnn_layer(cnn_model)
        cnn_model = self._cnn_layer(cnn_model)
        cnn_model = self._cnn_layer(cnn_model)

        x = Flatten(name='flatten')(cnn_model)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, name='embeded')(x)
        x = Lambda(self._l2_norm, output_shape=[512])(x)

        self.base_network = Model(inputs=input, outputs=x, name='finetuned_inception')
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

        self.model.compile(
            optimizer=Adam(),
            loss=self._contrastive_loss, 
            metrics=[self._accuracy]
        )


    def train(self):
        # TODO: fill parameters
        self.model.fit()


    def _l2_norm(self, x):
        return K.l2_normalize(x, axis=-1)


    def _contrastive_loss(self, positive_pair, y_pred):
        max_IQA = 100.0
        if positive_pair:
            return (2 / max_IQA) * y_pred**2
        else:
            return 2 * max_IQA * K.exp(-(2.77 * y_pred) / max_IQA)
    

    def _euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))


    def _eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)


    # def _accuracy(self, y_true, y_pred):
    #     '''Compute classification accuracy with a fixed threshold on distances.
    #     '''
    #     return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


    def _accuracy(self, _, y_pred):
        return K.mean(y_pred[:,0,0] < y_pred[:,1,0])


    def _cnn_layer(self, model):
        return Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=self.input_shape)(model)


    def _max_pool_layer(self, model):
        return MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model)
