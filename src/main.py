import os
import time

import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

import keras
import keras.callbacks as callbacks
from keras.preprocessing.image import ImageDataGenerator
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# project files
import utils.filehandler as filehandler
from models.FFN import FeedForwardNetwork as FFN
from models.lenet5 import LeNet5

callback_list = []

##################
#      DEBUG     #
##################
#ENV
debug = False

path = None
#creating our result directory
if(debug == False):
    path = "result/split_5" #+ str(time.time())
    os.makedirs(path, exist_ok=True)
    path += "/"
    callback_list.append(callbacks.ModelCheckpoint(
        path + "model.h5", 
        monitor='val_acc', 
        save_best_only=True))

##################
#    load data   #
##################

dataset = filehandler.import_csv_reshape('../fer2017/fer2017-training.csv')
(x_train, y_train, x_validation, y_validation) = filehandler.classic_split(dataset, 0.05)
#(x_train, y_train, x_validation, y_validation) = filehandler.balance_dataset(dataset, 0.45)

#and converting our values to categorical values
y_train = keras.utils.to_categorical(y_train, 10)
y_validation = keras.utils.to_categorical(y_validation, 10)

#normalisation
x_train = x_train * 1./255
x_validation = x_validation * 1./255





####################
# model parameters #
####################

loss_funct = 'categorical_crossentropy'
optimizer = 'adam'
size_x = x_train.shape[1]
size_y = x_train.shape[2]
input_shape = (size_x, size_y, 1) #grayscale
number_classes = 10
nb_epoch=200
batch_size = 128

train_gen = ImageDataGenerator(
    rotation_range=60,
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    brightness_range=None, 
    shear_range=0.2, 
    zoom_range=0.0, 
    fill_mode='nearest', 
    horizontal_flip=True, 
    vertical_flip=False, 
    rescale=None, 
    preprocessing_function=None
)
valid_gen = ImageDataGenerator()

#creating callbacks:
#callback_list.append(callbacks.EarlyStopping(min_delta=0.1, patience=20, verbose=1))
#callback_list.append(callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_delta=0.01, cooldown=0, min_lr=0))




################################
#            train             #
################################

#model = FFN(train_gen, valid_gen, nb_epoch, batch_size, debug=1)
model = LeNet5(train_gen, valid_gen, nb_epoch, batch_size, debug=1)
#model.build_model(input_shape, optimizer, loss_funct, number_classes, layer0_size=1000, layer1_size=512)
model.build_model(input_shape, optimizer, loss_funct, number_classes)

history = model.train(x_train, y_train, x_validation, y_validation, callback_list)




################################
#          saving              #
################################

val_accuracy = history['val_acc']
val_loss = history['val_loss']
accuracy = history['acc']
loss = history['loss']

if(path is not None):
    plt.plot(val_loss, label="validation loss")
    plt.plot(loss, label="train loss")
    plt.title("Loss evolution")
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.legend()
    plt.savefig(path + "loss.png")
    plt.clf()

    plt.plot(val_accuracy, label="validation accuracy")
    plt.plot(accuracy, label="train accuracy")
    plt.title("Accuracy evolution ")
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.legend()
    plt.savefig(path + "accuracy.png")
    plt.clf()