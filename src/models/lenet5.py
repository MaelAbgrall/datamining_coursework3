import multiprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy
class LeNet5():
    def __init__(self, train_gen, valid_gen, nb_epoch, batch_size, debug=0):

        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size

        self.debug = debug

    def build_model(self, input_shape, optimizer, loss_funct, number_classes):

        lenet_model = Sequential()
        lenet_model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        lenet_model.add(Activation('relu'))
        lenet_model.add(Conv2D(32, (3, 3)))
        lenet_model.add(Activation('relu'))
        lenet_model.add(MaxPooling2D(pool_size=(2, 2)))
        lenet_model.add(Dropout(0.25))

        lenet_model.add(Conv2D(64, (3, 3), padding='same'))
        lenet_model.add(Activation('relu'))
        lenet_model.add(Conv2D(64, (3, 3)))
        lenet_model.add(Activation('relu'))
        lenet_model.add(MaxPooling2D(pool_size=(2, 2)))
        lenet_model.add(Dropout(0.25))

        lenet_model.add(Flatten())
        lenet_model.add(Dense(512))
        lenet_model.add(Activation('relu'))
        lenet_model.add(Dropout(0.5))
        lenet_model.add(Dense(number_classes))
        lenet_model.add(Activation('softmax'))

        lenet_model.compile(loss=loss_funct, optimizer=optimizer, metrics=['accuracy'])

        self.model = lenet_model
        
 
    def train(self, x_train, y_train, x_validation, y_validation, callback_list):
        data_size = x_train.shape[0]
        print("training on ", data_size, "images")

        history_cbk = self.model.fit_generator(
            self.train_gen.flow(x_train, y_train, batch_size=self.batch_size),
            epochs=self.nb_epoch,
            steps_per_epoch= data_size / self.batch_size, 
            validation_data=(x_validation, y_validation),
            workers=multiprocessing.cpu_count(),
            verbose=self.debug,
            callbacks=callback_list)

        return history_cbk.history
