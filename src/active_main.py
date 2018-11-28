import os
import time

import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


import keras
from keras.preprocessing.image import ImageDataGenerator
import keras.callbacks as callbacks
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# project files
import utils.filehandler as filehandler
from models.lenet5_activeleraning import LeNet5

callback_list = []

##################
#      DEBUG     #
##################
debug = False

path = None
#creating our result directory
if(debug == False):
    path = "result/active_MS_128_" + str(time.time())
    os.makedirs(path, exist_ok=True)
    path += "/"
    callback_list.append(callbacks.ModelCheckpoint(
        path + "model.h5", 
        monitor='val_acc', 
        save_best_only=True))



#######################
#      Query Type     #
#######################
#query type
#    least confidence
#query_type = 'LC'
#    margin sampling
query_type = 'MS'
#    entropy
#query_type = 'EN'

#query_type = 'AD'


##################
#    load data   #
##################
#loading our dataset
dataset = filehandler.import_csv_reshape('../fer2017/fer2017-training.csv')
(x_val1, y_val1, x_val2, y_val2) = filehandler.classic_split(dataset, 0.75)
#concatenating (we only need one big validation dataset)
x_validation = numpy.concatenate((x_val1, x_val2), axis=0)
y_validation = numpy.concatenate((y_val1, y_val2), axis=0)

#and converting our values to categorical values
y_validation = keras.utils.to_categorical(y_validation, 10)

# adding a color channel to those grayscale images
#x_validation = numpy.expand_dims(x_validation, axis=3)

#normalisation
x_validation = x_validation * 1./255

# labelisation of the 10 first images:
x_train = []
y_train = []
for position in range(128):
    #shift the image and the label to the train set
    x_train.append(x_validation[position])
    y_train.append(y_validation[position])
    #removing this element from the validation set
    x_validation = numpy.delete(x_validation, position, axis=0)
    y_validation = numpy.delete(y_validation, position, axis=0)

x_train = numpy.array(x_train)
y_train = numpy.array(y_train)




####################
# model parameters #
####################

# model parameters
loss_funct = 'categorical_crossentropy'
optimizer = 'adam'
size_x = x_train.shape[1]
size_y = x_train.shape[2]
input_shape = (size_x, size_y, 1) #grayscale
number_classes = 10
step_epoch = 20 #number of epochs for a step
batch_size = 128

#data generator
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




################################
#            train             #
################################
#building model
lenet = LeNet5(train_gen, valid_gen, step_epoch, batch_size, query_type=query_type, path=path)
lenet.build_model(input_shape, optimizer, loss_funct, number_classes)

#training
(val_acc, val_loss, acc, loss) = lenet.train(x_train, y_train, x_validation, y_validation, callback_list)


################################
#          saving              #
################################

if(debug == False):
    plt.plot(val_loss, label="validation loss")
    plt.plot(loss, label="train loss")
    plt.title('Loss evolution')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.legend()
    plt.savefig(path + "loss.png")

    plt.plot(val_acc, label="validation accuracy")
    plt.plot(acc, label="train accuracy")
    plt.title('Accuracy evolution')
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.legend()
    plt.savefig(path + "accuracy.png")
