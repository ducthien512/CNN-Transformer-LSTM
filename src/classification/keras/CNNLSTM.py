import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, MaxPool2D, SpatialDropout2D, BatchNormalization, LSTM
from tensorflow.keras.layers import Flatten, InputSpec, Layer, Concatenate, AveragePooling2D, MaxPooling2D, Reshape, Permute
from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D, LayerNormalization, SeparableConv1D, MaxPooling1D
from tensorflow.keras.layers import TimeDistributed, Lambda, AveragePooling1D, Add, Conv1D, Multiply, DepthwiseConv1D
from tensorflow.keras.constraints import max_norm, unit_norm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from keras import Model, Sequential
from keras.constraints import max_norm
from keras.layers import Input,Conv2D, BatchNormalization, Dropout, AveragePooling2D, Flatten, Dense, DepthwiseConv2D, \
    Activation, SeparableConv2D, Conv1D, DepthwiseConv1D, SeparableConv1D, AveragePooling1D

from classification.keras.KerasClassifier import KerasClassifier
from config.Config import config
from preprocessing.DataRepresentation import DataRepresentation
from preprocessing.NDScaler import NDScaler
from utils import file_manager


class CNNLSTM(KerasClassifier):

    def _create_model(self, input_shape: tuple, num_classes: int) -> Model:
        height = input_shape[0]
        conv_depth = 2
        x_input = Input(shape=input_shape)

        if config.data_representation == DataRepresentation.TIME_FREQUENCY:

            # CNN blocks
            x = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x_input)
            x = BatchNormalization()(x)
            x = DepthwiseConv2D((height, 1), depth_multiplier=conv_depth, depthwise_constraint=max_norm(1.))(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((1, 4))(x)

            x = Conv2D(filters=128, kernel_size=(7, 7), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((1, 4))(x)

            x = Conv2D(filters=128, kernel_size=(7, 7), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((1, 4))(x)

            cnn_output = Dropout(0.5)(x)

        else:
            # CNN blocks
            x = Conv1D(64, kernel_size=7, strides=1, padding="same", kernel_initializer="he_normal")(x_input)
            x = BatchNormalization()(x)
            x = DepthwiseConv1D(64, depth_multiplier=conv_depth, depthwise_constraint=max_norm(1.))(x)
            x = Activation('relu')(x)
            x = MaxPooling1D(4)(x)

            x = Conv1D(128, kernel_size=7, strides=1, padding="same", kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling1D(4)(x)

            x = Conv1D(128, kernel_size=7, strides=1, padding="same", kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling1D(4)(x)

            cnn_output = Dropout(0.5)(x)

        # Reshape cnn_output
        lstm_input = Reshape((-1, cnn_output.shape[-1]))(cnn_output)

        # LSTM block
        lstm_output = LSTM(units=128, dropout=0.5, activation='tanh', return_sequences=True)(lstm_input)

        x = Flatten()(lstm_output)
        x = Dense(128, activation='relu')(x)

        # Output layer
        output = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=x_input, outputs=output, name=self.get_name())

        model.summary()
        plot_model(model, "model.png", show_shapes=True)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def get_name(self) -> str:
        return 'CNNLSTM'

    def _preprocess_dataset(self, data: tuple, augmentation_method_name: str) -> tuple:
        (x_train, y_train), (x_test, y_test) = super()._preprocess_dataset(data, augmentation_method_name)

        object_type = 'scaler'
        scaler = file_manager.load_pickle_object(self.get_name(), object_type, augmentation_method_name)
        if scaler:
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
        else:
            scaler = NDScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            file_manager.save_pickle_object(self.get_name(), object_type, scaler, augmentation_method_name)

        if config.data_representation == DataRepresentation.TIME_FREQUENCY:
            # Transform the input from n_epochs, n_channels, n_frequency/(height), n_times (width)
            #                     to   n_epochs, n_frequency(height), n_times (width), n_channels
            # in order to match the input shapes of the Conv2D
            x_train = np.transpose(x_train, (0, 2, 3, 1))
            x_test = np.transpose(x_test, (0, 2, 3, 1))
        else:
            x_train = np.transpose(x_train, (0, 2, 1))
            x_test = np.transpose(x_test, (0, 2, 1))

        return (x_train, y_train), (x_test, y_test)
