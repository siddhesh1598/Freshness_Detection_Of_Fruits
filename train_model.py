# USAGE
# python train.py --dataset dataset --model pokedex.model --labelbin lb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from imutils import paths

import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import pickle
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", type=str, default="lb.pickle",
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize parameters
lr = 1e-3
epochs = 50
batch_size = 32

# load dataset
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# iterate over all images and append
# images to data list
# labels to labels list
for imagePath in imagePaths:
	label = imagePath.split(os.path.sep)[-2]

	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))

	data.append(image)
	labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# encoding the dataset
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition dataset into training and testing
(X_train, X_test, y_train, y_test) = train_test_split(data,
										labels,
										test_size=0.2,
										random_state=42)

# model
vgg = VGG16(weights="imagenet", include_top=False,
			input_tensor=Input(shape=(224, 224, 3)))

model = vgg.output
model = AveragePooling2D(pool_size=(4, 4))(model)
model = Flatten()(model)
model = Dense(64, activation="relu")(model)
model = Dropout(0.5)(model)
model = Dense(len(lb.classes_), activation="softmax")(model)

model = Model(inputs=vgg.input, outputs=model)

for layer in vgg.layers:
	layer.trainable = False

# compile
print("[INFO] compiling model...")
opt = Adam(lr=lr, decay=lr/epochs)
model.compile(loss="categorical_crossentropy",
				optimizer=opt,
				metrics=["accuracy"])
'''
# train
print("[INFO] training model...")
H = model.fit(
	x=X_train,
	y=y_train,
	batch_size=batch_size,
	steps_per_epoch=len(X_train)//batch_size,
	validation_data=(X_test, y_test),
	validation_steps=len(X_test)//batch_size,
	epochs=epochs)
'''

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# train the network
print("[INFO] training network...")
H = model.fit(
	aug.flow(X_train, y_train, batch_size=batch_size),
	validation_data=(X_test, y_test),
	steps_per_epoch=len(X_train) // batch_size,
	epochs=epochs, verbose=1)

# test
print("[INFO] evaluating model...")
predIdxs = model.predict(X_test, batch_size=batch_size)
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(y_test.argmax(axis=1),
						predIdxs,
						target_names=lb.classes_))

# confusion matrix
cm = confusion_matrix(y_test.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

# save model
print("[INFO] saving COVID-19 detector model...")
model.save(args["model"], save_format="h5")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
