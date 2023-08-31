import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ydata_profiling as pp
import numpy as np
from sklearn.metrics import accuracy_score,classification_report,f1_score, precision_recall_curve

import tensorflow as tf
from keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,BatchNormalization,Flatten,Conv2D,MaxPooling2D,Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import TensorBoard

import warnings
warnings.filterwarnings('ignore')

epochs = 40
la1=pd.read_csv("C:\\Users\\preet\\Desktop\\Python\\ML_Tasks\\miml_dataset\\miml_labels_1.csv")
image_directory = "C:\\Users\\preet\\Desktop\\Python\\ML_Tasks\\miml_dataset\\images"
test_directory = "C:\\Users\\preet\\Desktop\\Python\\ML_Tasks\\miml_dataset\\test"
class_list = list(la1.columns[1:])
print(class_list)

train_datagen=ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.5,
    horizontal_flip=True,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    fill_mode="nearest",
    rescale=1./255,
    validation_split=0.2,
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_dataframe(
    dataframe=la1[:1800],
    directory=image_directory,
    x_col="Filenames",
    y_col=class_list,
    batch_size=32,
    shuffle=True,
    class_mode="raw",
    target_size=(128,128),
    classes=class_list
)

valid_generator=test_datagen.flow_from_dataframe(
    dataframe=la1[1800:],
    directory=image_directory,
    x_col="Filenames",
    y_col=class_list,
    batch_size=32,
    shuffle=True,
    class_mode="raw",
    target_size=(128,128),
    classes=class_list
)

@tf.__internal__.dispatch.add_dispatch_support
def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1):

    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)
    label_smoothing = tf.convert_to_tensor(label_smoothing, dtype=y_pred.dtype)
    y_true = tf.__internal__.smart_cond.smart_cond(label_smoothing, lambda: y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing, lambda: y_true)
    return backend.mean(backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits),axis=axis)

#inputs = Input(shape=(128,128,3))

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(
        32, (3,3), activation="relu", input_shape= (128,128,3)
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(
        32,(3,3), activation="relu"
    ),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(
        32,(3,3), activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(5, activation="softmax")

])

model.compile(optimizer=Adamax(),
              loss=binary_crossentropy,
              metrics=['accuracy', 'binary_accuracy']
)
model.summary()

tensorboard_callback = TensorBoard(log_dir='Synapse_logs', histogram_freq=1)
model.fit_generator(train_generator,validation_data=valid_generator,epochs=epochs,callbacks=tensorboard_callback)

#model.save("")

def evaluate_model(model, generator):
    true_labels = generator.labels  # True labels
    y_pred_proba = model.predict(generator)  # Predicted probabilities
    precision, recall, thresholds = precision_recall_curve(true_labels.ravel(),
                                                           y_pred_proba.ravel())  # Optimal threshold found

    # Calculate F1 Scores
    f1_scores = (2*precision*recall)/(precision+recall)

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    print('Optimal threshold:', optimal_threshold)

    y_pred = (y_pred_proba > 0.5).astype(int)
    f1 = f1_score(true_labels, y_pred, average='macro')
    print('F1 score:', f1)

    y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int)
    return y_pred_optimal


def show_image_prediction(generator, model, class_list):
    y_pred_optimal = evaluate_model(model, generator)
    x, y_true = generator[0]
    img = (x * 255).astype('uint8')  # Scale back the image
    true_labels = y_true[0]
    true_labels_str = [class_list[i] for i in range(len(true_labels)) if true_labels[i] == 1]  # Decode to string labels
    pred_labels = y_pred_optimal[0]

    pred_labels_str = [class_list[i] for i in range(len(pred_labels)) if pred_labels[i] == 1]

    plt.imshow(img[0])
    print('True labels:', true_labels_str)
    print('Predicted labels:', pred_labels_str)

show_image_prediction(valid_generator, model, class_list)








"""
    tf.keras.layers.Conv2D(
        32, (3,3), activation="relu", input_shape= (128,128,3)
    ),
    #tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(
        32,(3,3), activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(5, activation="softmax")
"""


