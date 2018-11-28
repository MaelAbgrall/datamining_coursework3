import multiprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import oracle

class LeNet5():
    def __init__(self, train_gen, valid_gen, step_epoch, batch_size, query_type=None, path=None): #TODO: add stop criterion -all images or overfit
        #TODO: add also a number of labelised images 

        self.train_gen = train_gen
        self.valid_gen = valid_gen
        self.step_epoch = step_epoch
        self.batch_size = batch_size

        #stop criterion
        self.stop_criterion = False

        self.path = path

        #query type
        self.query_type = query_type
        if(query_type == None):
            print("no query type selected!")

        #history
        self.history_val_loss = []
        self.history_val_accuracy = []
        self.history_loss = []
        self.history_accuracy = []

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
        
    def _step_train(self, x_train, y_train, x_validation, y_validation):
        """
        do a training on the given data and return the history
        """

        data_size = x_train.shape[0]
        print("training on ", data_size, "images")

        history_cbk = self.model.fit_generator(
            self.train_gen.flow(x_train, y_train, batch_size=self.batch_size),
            epochs=self.step_epoch,
            steps_per_epoch= data_size / self.step_epoch, 
            validation_data=(x_validation, y_validation),
            workers=multiprocessing.cpu_count(),
            verbose=0,
            callbacks = self.callbacks)
        return history_cbk.history

    def _labelise(self, x_train, y_train, x_validation, y_validation):
        #finding the most "useful" data
        predictions = self.model.predict(x_validation, batch_size=self.batch_size)
        
        if(self.query_type == 'LC'):
            image_indices = oracle.least_confidence(predictions)

        if(self.query_type == 'MS'):
            image_indices = oracle.margin_sampling(predictions)

        if(self.query_type == 'EN'):
            image_indices = oracle.entropy(predictions)

        if(self.query_type == 'AD'):
            image_indices = oracle.adversarial_margin(self.model, x_validation, y_validation)

        # add our images to the train set
        x_train = numpy.append(x_train, x_validation[image_indices], axis=0)
        y_train = numpy.append(y_train, y_validation[image_indices], axis=0)
        # remove them from the validation set
        x_validation = numpy.delete(x_validation, image_indices, axis=0)
        y_validation = numpy.delete(y_validation, image_indices, axis=0)

        return (x_train, y_train, x_validation, y_validation)

    def _evaluate_stop(self, y_train, y_validation):
        """
        if 75% of the dataset is labeled we stop training
        """
        train_size = y_train.shape[0]
        validation_size = y_validation.shape[0]
        max_labelled = int( (train_size + validation_size) * 0.75 )

        if(train_size > max_labelled):
            self.stop_criterion = True

    def train(self, x_train, y_train, x_validation, y_validation, callback_list):
        self.callbacks = callback_list
        
        #first training
        step_history = self._step_train(x_train, y_train, x_validation, y_validation)

        #saving global history
        self.history_val_accuracy.extend(step_history['val_acc'])
        self.history_val_loss.extend(step_history['val_loss'])
        self.history_accuracy.extend(step_history['acc'])
        self.history_loss.extend(step_history['loss'])

        if(self.path is not None):
            plt.plot(self.history_val_loss, label="validation loss")
            plt.plot(self.history_loss, label="train loss")
            plt.title("Loss evolution for " + str(y_train.shape[0]) + " images")
            plt.ylabel('Loss')
            plt.xlabel('Iterations')
            plt.legend()
            plt.savefig(self.path + "loss" + str(y_train.shape[0]) + ".png")
            plt.clf()

            plt.plot(self.history_val_accuracy, label="validation accuracy")
            plt.plot(self.history_accuracy, label="train accuracy")
            plt.title("Accuracy evolution for " + str(y_train.shape[0]) + " images")
            plt.ylabel('Accuracy')
            plt.xlabel('Iterations')
            plt.legend()
            plt.savefig(self.path + "accuracy" + str(y_train.shape[0]) + ".png")
            plt.clf()

        #labelisation of more images
        (x_train, y_train, x_validation, y_validation) = self._labelise(x_train, y_train, x_validation, y_validation)

        while(self.stop_criterion == False):
            #training
            step_history = self._step_train(x_train, y_train, x_validation, y_validation)

            #saving global history
            self.history_val_accuracy.extend(step_history['val_acc'])
            self.history_val_loss.extend(step_history['val_loss'])
            self.history_accuracy.extend(step_history['acc'])
            self.history_loss.extend(step_history['loss'])

            if(self.path is not None):
                plt.plot(self.history_val_loss, label="validation loss")
                plt.plot(self.history_loss, label="train loss")
                plt.title("Loss evolution for " + str(y_train.shape[0]) + " images")
                plt.ylabel('Loss')
                plt.xlabel('Iterations')
                plt.legend()
                plt.savefig(self.path + "loss" + str(y_train.shape[0]) + ".png")
                plt.clf()

                plt.plot(self.history_val_accuracy, label="validation accuracy")
                plt.plot(self.history_accuracy, label="train accuracy")
                plt.title("Accuracy evolution for " + str(y_train.shape[0]) + " images")
                plt.ylabel('Accuracy')
                plt.xlabel('Iterations')
                plt.legend()
                plt.savefig(self.path + "accuracy" + str(y_train.shape[0]) + ".png")
                plt.clf()

            #labelisation of the data. If there is no data left to labelise, the program will stop
            (x_train, y_train, x_validation, y_validation) = self._labelise(x_train, y_train, x_validation, y_validation)            

            #if we are training until a certain level of accuracy (and not until there is no label left)
            self._evaluate_stop(y_train, y_validation)

        return (self.history_val_accuracy, self.history_val_loss, self.history_accuracy, self.history_loss)