from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD, Adamax, Nadam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
import keras.backend as K
from keras.preprocessing import image
from keras.models import load_model
from keras_tqdm import TQDMNotebookCallback

import json
import os
import numpy as np
import pandas as pd
from Generator import DriveDataGenerator
from Cooking import checkAndCreateDir
import h5py
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# << The directory containing the cooked data from the previous step >>
COOKED_DATA_DIR = 'cooked_data'

# << The directory in which the model output will be placed >>
MODEL_OUTPUT_DIR = 'models'


train_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'train.h5'), 'r')
eval_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'eval.h5'), 'r')

num_train_examples = train_dataset['image'].shape[0]
print(num_train_examples)
num_eval_examples = eval_dataset['image'].shape[0]
print(num_eval_examples)

batch_size = 32

data_generator = DriveDataGenerator(rescale=1. / 255., brighten_range=0.4)
train_generator = data_generator.flow \
    (train_dataset['image'], train_dataset['previous_state'], train_dataset['label'], batch_size=batch_size,
     zero_drop_percentage=0.95, roi=[0, 143, 0, 255])
eval_generator = data_generator.flow \
    (eval_dataset['image'], eval_dataset['previous_state'], eval_dataset['label'], batch_size=batch_size,
     zero_drop_percentage=0.95, roi=[0, 143, 0, 255])

def hanging_line(point1, point2):
    a = (point2[1] - point1[1])/(np.cosh(point2[0]) - np.cosh(point1[0]))
    b = point1[1] - a*np.cosh(point1[0])
    x = np.linspace(point1[0],point2[0],80)
    y = np.linspace(point1[1],point2[1],80)
    y_add_former=np.sqrt(np.linspace(0,100,40)).tolist()
    y_add_later = np.sqrt(np.linspace(100, 0, 40)).tolist()
    y_add=y_add_former+y_add_later
    y = y-y_add
    return (x,y)

def draw_image_with_label(img, label, prediction=None):
    theta = label * 0.4  # Steering range for the car is +- 40 degrees -> 0.69 radians
    line_length = 50
    line_thickness = 3
    label_line_color = (255, 0, 0)
    prediction_line_color = (0, 0, 255)
    pil_image = image.array_to_img(img, K.image_data_format(), scale=True)
    print('Actual Steering Angle = {0}'.format(label))
    draw_image = pil_image.copy()
    image_draw = ImageDraw.Draw(draw_image)
    first_point = (int(img.shape[1] / 2), img.shape[0])
    second_point = (
    int((img.shape[1] / 2) + (line_length * math.sin(theta))), int(img.shape[0] - (line_length * math.cos(theta))))
    #print(first_point,second_point)
    x,y=hanging_line(first_point,second_point)
    #plt.plot((0,100), 'o')
    #image_draw.line([first_point, second_point], fill=label_line_color, width=line_thickness, joint='curve')
    #plt.plot(x,y,'o','--r',markersize=1)
    style = "Simple,tail_width=1,head_width=4,head_length=8"
    kw = dict(arrowstyle=style, color="red")
    kw_str="arc3,rad="+str(float(-theta)/2)
    a = patches.FancyArrowPatch(first_point, second_point, connectionstyle=kw_str, **kw)
    plt.gca().add_patch(a)

    if (prediction is not None):
        print('Predicted Steering Angle = {0}'.format(prediction))
        print('L1 Error: {0}'.format(abs(prediction - label)))
        theta = prediction * 0.4
        second_point = (
        int((img.shape[1] / 2) + (line_length * math.sin(theta))), int(img.shape[0] - (line_length * math.cos(theta))))
        #image_draw.line([first_point, second_point], fill=prediction_line_color, width=line_thickness)
        style = "Simple,tail_width=1,head_width=4,head_length=8"
        kw = dict(arrowstyle=style, color="blue")
        kw_str = "arc3,rad=" + str(float(-theta) / 2)
        a = patches.FancyArrowPatch(first_point, second_point, connectionstyle=kw_str, **kw)
        plt.gca().add_patch(a)

    del image_draw
    plt.imshow(draw_image)
    plt.show()

MODEL_PATH = './models/fresh_models/model_model.326-0.0008444.h5' # model_model.101-0.0061546.h5
model = load_model(MODEL_PATH)

[sample_batch_train_data, labels] = next(train_generator)

predictions = model.predict([sample_batch_train_data[0]])

print(labels,predictions)

for i in range(0, 10, 1):
    print(labels[i],predictions[i])
    draw_image_with_label(sample_batch_train_data[0][i], labels[i], predictions[i])

