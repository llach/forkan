from keras.applications import InceptionResNetV2, Xception, NASNetLarge, VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint

from forkan import weights_path
from forkan.datasets.image import load_image_dataset

import sys
import logging

logger = logging.getLogger(__name__)
model_checkpoint_path = weights_path + '/object_rec/'

MODEL2ABR = {
    'inceptionresnetv2': 'IR2',
    'xception': 'XC',
    'nasnetlarge': 'NNL',
    'vgg19': 'vgg19',
}

MODEL2CLASS = {
    'inceptionresnetv2': InceptionResNetV2,
    'xception': Xception,
    'nasnetlarge': NASNetLarge,
    'vgg19': VGG19,
}

MODEL_NAME = 'nasnetlarge'

DATASET_NAME = 'drinks8'
TARGET_SIZE = [240, 240, 3]

EPOCHS = 50
BATCH_SIZE = 32

if MODEL_NAME not in MODEL2ABR:
    logger.critical('Model {} not available!'.format(MODEL_NAME))
    sys.exit(1)

# load and unpack dataset
train, test, mappings = load_image_dataset(DATASET_NAME)

x_train, y_train = train
x_test, y_test = test
idx2label, label2idx = mappings
idx2label = idx2label.item()
label2idx = label2idx.item()

# get number of classes
nb_classes = len(idx2label.keys())

# input based on out dataset
input_tensor = Input(shape=x_train.shape[1:])

# create the base pre-trained model
if MODEL_NAME is 'nasnetlarge':
    # NASNETLarge weights without top has a bug when loading,
    # so we load it with top layers and remove them afterwards
    base_model = MODEL2CLASS[MODEL_NAME](input_tensor=input_tensor, weights='imagenet', include_top=True)

    base_model.layers.pop()
    base_model.layers.pop()

else:
    base_model = MODEL2CLASS[MODEL_NAME](input_tensor=input_tensor, weights='imagenet', include_top=False)


# don't train the base layers we've just loaded
for layer in base_model.layers:
    layer.trainable = False

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(base_model.output)

# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# add some dropout
x = Dropout(.25)(x)

# and a logistic layer
predictions = Dense(nb_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',  metrics=['accuracy'])

# check our final model
model.summary()

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    fill_mode='nearest',
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)

test_datagen = ImageDataGenerator(
    horizontal_flip=True,
    fill_mode='nearest',
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)

# fit generators to data
train_datagen.fit(x_train)
test_datagen.fit(x_test)

# Save the model according to the conditions
checkpoint = ModelCheckpoint(model_checkpoint_path + MODEL2ABR[MODEL_NAME] + '_' + DATASET_NAME +
                             '_E{epoch}_VA{val_acc:.2f}.hdf5',monitor='val_acc', verbose=1,save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)

# Train the model
model.fit_generator(
    train_datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=(x_train.shape[0]),
    epochs=EPOCHS,
    validation_data=test_datagen.flow(x_test, y_test),
    callbacks=[checkpoint])
