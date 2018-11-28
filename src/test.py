import os
import time
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # only use cpu
import keras
import numpy

# project files
import utils.filehandler as filehandler

from keras.models import load_model

path=None

if(len(sys.argv) >= 2):
    path = sys.argv[1]

if(len(sys.argv) == 1):
    raise Exception("please enter a folder path where model.h5 is saved\nex: ./result/myfolder")



# load model
model = load_model( path + '/model.h5')

# load data
dataset = filehandler.import_csv_reshape('../fer2017/fer2017-testing.csv')

# split data
(x_test, y_test, _, _) = filehandler.classic_split(dataset, 1.)

# preprocessing
x_test = x_test * 1./255


# predict
predictions = model.predict(x_test)

# evaluate
class_output = numpy.argmax(predictions, axis=1)
correct = 0
total = 0
for position in range(class_output.shape[0]):
    total += 1
    if(class_output[position] == y_test[position]):
        correct +=1
        
        
"correct:...", correct, "\ntotal:....", total, "\n%: ", correct/total

eval_string = "correct:... " + str(correct) + "\ntotal:..... " + str(total) + "\n%:......... " + str(correct/total)
print(eval_string)

with open(path + "/test_score.txt", "w+") as text_file:
        text_file.write(eval_string)