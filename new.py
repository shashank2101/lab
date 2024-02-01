from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, MaxPooling3D, Flatten, Dropout, Dense, LeakyReLU, Concatenate
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import DirectoryIterator
import numpy as np
class Classifier:
    def __init__(self):
        self.model = 0

    def predict(self, x):
        if x.size == 0:
            return []
        return self.model.predict(x)

    def fit(self, x=None, y=None, epochs=None, validation_data=None):
        if isinstance(x, DirectoryIterator):
            # Print information about the provided data generator
            print("Using data generator for training.")
            print("Number of batches:", len(x))
            print("Batch size:", x.batch_size)
            print("Target size of images:", x.image_shape)
            print("Using a DirectoryIterator. Fetching the first batch to inspect shapes.")
            # Check the first batch to see the shape of data and labels
            sample_batch = next(iter(x))
            print("Shape of a sample batch (data, labels):", sample_batch[0].shape, sample_batch[1].shape)
            return self.model.fit(x, epochs=epochs, validation_data=validation_data)
        elif x is not None and y is not None:
            print("Shapes of input data and labels:")
            x = np.expand_dims(x, axis=2)
            x = np.repeat(x, 256, axis=2)
            print("Input data shape:", x.shape)

            print("Labels shape:", y.shape)
            return self.model.fit(x, y, epochs=epochs, validation_data=validation_data)
        else:
            raise ValueError("Either both 'x' and 'y' or a data generator must be provided.")
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)

    def evaluate(self, x=None, y=None):
        if x is not None and y is not None:
            return self.model.evaluate(x, y)
        elif isinstance(x, DirectoryIterator):
            # If a data generator is provided, use it directly
            return self.model.evaluate(x)
        else:
            raise ValueError("Either both 'x' and 'y' or a data generator must be provided.")

class MesoInception4_3D(Classifier):
    def __init__(self, learning_rate=0.001, num_frames=16):
        self.model = self.init_model(num_frames)
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    def InceptionLayer(self, a, b, c, d):
        def func(x):
            x1 = Conv3D(a, (1, 1, 1), padding='same', activation='relu')(x)

            x2 = Conv3D(b, (1, 1, 1), padding='same', activation='relu')(x)
            x2 = Conv3D(b, (3, 3, 3), padding='same', activation='relu')(x2)

            x3 = Conv3D(c, (1, 1, 1), padding='same', activation='relu')(x)
            x3 = Conv3D(c, (3, 3, 3), dilation_rate=2, strides=1, padding='same', activation='relu')(x3)

            x4 = Conv3D(d, (1, 1, 1), padding='same', activation='relu')(x)
            x4 = Conv3D(d, (3, 3, 3), dilation_rate=3, strides=1, padding='same', activation='relu')(x4)

            y = Concatenate(axis=-1)([x1, x2, x3, x4])

            return y

        return func

    def init_model(self, num_frames):
        x = Input(shape=(num_frames, 256, 256, 3))
        print("Input Shape:", x.shape)

        x1 = self.InceptionLayer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling3D(pool_size=(1, 2, 2), padding='same', trainable=False)(x1)
        print("After InceptionLayer and MaxPooling3D Shape:", x1.shape)

        x2 = self.InceptionLayer(2, 4, 4, 2)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling3D(pool_size=(1, 2, 2), padding='same', trainable=False)(x2)
        print("After Second InceptionLayer and MaxPooling3D Shape:", x2.shape)

        x3 = Conv3D(16, (1, 5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling3D(pool_size=(1, 2, 2), padding='same')(x3)
        print("After Conv3D and MaxPooling3D Shape:", x3.shape)

        x4 = Conv3D(16, (1, 5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling3D(pool_size=(4, 4, 4), padding='same')(x4)
        print("After Second Conv3D and MaxPooling3D Shape:", x4.shape)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        print("Final Output Shape:", y.shape)

        return KerasModel(inputs=x, outputs=y)
