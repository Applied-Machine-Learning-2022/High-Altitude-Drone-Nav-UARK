import tensorflow
import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import pandas as pd
import math
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.optimizer_v1 import SGD
import pathlib
import pickle
from PIL import Image
import tensorflow_probability as tfp
import sys
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from keras.applications.xception import Xception
from tensorflow.python.keras.callbacks import TerminateOnNaN
from pathlib import Path

# Previously when I had very big images this was needed to prevent a crash, no longer necessary but doesn't hurt
Image.MAX_IMAGE_PIXELS = 1000000000

# This can enhance speed by decreasing floating point precision, optional
# There are some additional tweaks you may need to do if you go this route, look it up on google
# keras.backend.set_floatx('float16')

# tf.config.run_functions_eagerly(True)
# This puts the path of the current directory of the python file as the source
workingpath = pathlib.Path.cwd()

# These are the arguments given when running the file
# Keep in mind if you want to run this file in pycharm you need to set these flags in the run configuration
opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

# Name of the test, for saving checkpoints
testname = args[0]
# String used to determine the test dataset: must be a consistent and unique string at start of each test dataset image
testsetstring = args[1]
# Initial learning rate
initialLR = float(args[2])
# l1 regularization parameter on most layers
l1_reg = float(args[3])
# l2 regularization parameter on most layers
l2_reg = float(args[4])
# Training run length
epochs = int(args[5])

distribution_type = str(args[6])

dropout = float(args[7])

p = pathlib.Path(workingpath / f'{testname:s}')
p.mkdir(parents=True, exist_ok=True)

# This indicates that the data should be found one directory prior to the directory the file is run from
# Simply change to the correct directory if not
traindatapath = "/Users/santiagodorado/Documents/Project Images 2/ProjImages" #str(workingpath / '..' )
testdatapath = "/Users/santiagodorado/Documents/Project Images 2/ProjImages" #str(workingpath / '..' )
#Location of the labels file: the file should be a csv with  filename, x, and y columns
alldata = pd.read_csv('/Users/santiagodorado/Downloads/GeolocalizationZipped 4/Code/multiplesourcelabels.csv')


# Divides out the test dataset from the train dataset
testmatch = [x.startswith(testsetstring) for x in alldata['Filename'].values]
trainmatch = [not i for i in testmatch]
train = alldata[trainmatch].copy()
test = alldata[testmatch].copy()
# test = alldata[trainmatch].copy()

# Coordinates of RoI from ArcGIS, the plus 1500 meters due to the fact that the coords are upper left corners of 3k chips (but arcgis samples as if from center)
topRoi = 4005539.977963 + 1500
bottomRoi = 3971320.265500 + 1500
leftRoi = 380030.074942 - 1500
rightRoi = 414249.787405 - 1500

heightRoi = topRoi - bottomRoi
widthRoi = rightRoi - leftRoi

centerY = (topRoi + bottomRoi) / 2.0
centerX = (rightRoi + leftRoi) / 2.0

sideWidth = 1.0

# Bounds for size restriction experiment, training set
topBound = centerY + sideWidth / 2.0 * heightRoi
bottomBound = centerY - sideWidth / 2.0 * heightRoi
leftBound = centerX - sideWidth / 2.0 * widthRoi
rightBound = centerX + sideWidth / 2.0 * widthRoi

# Train set restrictions
inareax = [((x > (leftBound)) and (x < (rightBound))) for x in train['x'].values]
train = train[inareax]
inareay = [((x > (bottomBound)) and (x < (topBound))) for x in train['y'].values]
train = train[inareay]

# Validation padding to avoid boundary effects
validPad = 1500

# Additional area restriction for test set only
inareax = [((x > (leftBound + validPad)) and (x < (rightBound - validPad))) for x in test['x'].values]
test = test[inareax]
inareay = [((x > (bottomBound + validPad)) and (x < (topBound - validPad))) for x in test['y'].values]
test = test[inareay]

# I additionally remove the ADOP2001 dataset from most runs due to it being false color infrared
remove = [x.startswith("CHIP_ADOP2001") for x in train['Filename'].values]
notremove = [not i for i in remove]
train = train[notremove].copy()

# This scaler will be used to scale the coordinates to between 0 and 1
# Scaling is based on ALL data, and is applied equally to train and test
# Now it is very important to make sure that, when we adjust labels during cropping, we don't screw things up with scaling
# Thus, we need to either combine the labels first THEN scale, or make sure to save just the scaling portion to scale the adjustment (current approach)
scaler = MinMaxScaler(feature_range=(0, 1))

alldata[['x', 'y']] = scaler.fit_transform(alldata[['x', 'y']])

# Scale parameters for x and y columns, for later use
scalexparam = scaler.scale_[0]
scaleyparam = scaler.scale_[1]

# print("scalex", 500* scalexparam)
# print(scaleyparam)

train[['x', 'y']] = scaler.transform(train[['x', 'y']])
# print(test.head(5))
test[['x', 'y']] = scaler.transform(test[['x', 'y']])
# print(test.head(5))




# In case you want the scaled labels of the test set in a file
# testx, testy = makeTransform(test, scalerx, scalery)
# numpy.savetxt("foo.csv", np.hstack((testx,testy)), delimiter=",")

# Old random crop code for generator based data input
def random_crop(img, random_crop_size, img_real_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :], (float(x) / img.shape[1]) * img_real_size, (float(y) / img.shape[0]) * img_real_size

# Generates crops, now replaced by map function below
def crop_generator(batches, crop_length, img_real_size, scalerx, scalery):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i], xshift, yshift = random_crop(batch_x[i], (crop_length, crop_length), img_real_size)
            # print(batch_y[i][1])
            xarray = np.array([batch_y[i][0]])
            yarray = np.array([batch_y[i][1]])

            # print(batch_y[i][0])
            # print (xarray[0])
            xarray[0] += xshift * scalexparam
            yarray[0] -= yshift * scaleyparam
            # xarray.reshape(1,-1)
            xarray = [[xarray[0]]]
            yarray = [[yarray[0]]]
            # print(xarray)
            # print(scalerx.transform(xarray))
            # print("TRANSFORMED", scalerx.transform(xarray)[0][0])
            batch_y[i][0], batch_y[i][1] = xarray[0][0], yarray[0][0]
            # print(batch_y[i])

        yield (batch_crops, batch_y)


def gaussian_kernel(x1, x2, beta=1.0):
    r = tf.transpose(x1)
    r = tf.expand_dims(r, 2)
    return tf.reduce_sum(tf.exp(-(1.0 / (2.0 * beta * beta)) * tf.square(r - x2)), axis=-1)


def MMD(x1, x2):
    """
    maximum mean discrepancy (MMD) based on Gaussian kernel
    function for keras models (theano or tensorflow backend)

    - Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
    Advances in neural information processing systems. 2007.
    """
    # bandwidth_range = [4, 8, 16, 32]
    # for x in bandwidth_range:

    # x1x1 = gaussian_kernel(x1, x1, beta)
    # print(x1x1)
    x1x1 = gaussian_kernel(x1, x1, 2.0)
    x1x2 = gaussian_kernel(x1, x2, 2.0)
    x2x2 = gaussian_kernel(x2, x2, 2.0)

    bandwidth_range = [4.0, 8.0, 16.0, 32.0, 64.0]
    for beta in bandwidth_range:
        x1x1 += gaussian_kernel(x1, x1, beta)
        # print(x1x1)
        x1x2 += gaussian_kernel(x1, x2, beta)
        # print(x1x2)
        x2x2 += gaussian_kernel(x2, x2, beta)

    diff = tf.reduce_mean(x1x1) - 2.0 * tf.reduce_mean(x1x2) + tf.reduce_mean(x2x2)
    return diff


# Replacement for random crop above: note that ROWS are our y coordinate, so y coordinate is FIRST in image matrix
# If we are making validation data, we always crop from the upper left (no adjustment of label), so there is no randomness
# Flag is currently not used (see validation versions of these functions below) due to not fully refactoring code

g1 = tf.random.Generator.from_non_deterministic_state()


def random_crop_single(img, random_crop_size, img_real_size):
    # Note: image_data_format is 'channel_last'
    # print("IMAGE SHAPE" , img.shape)
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]

    dy, dx = random_crop_size

    # x = np.random.randint(0, width - dx + 1)
    x = g1.uniform(shape=(), minval=0, maxval=width - dx + 1, dtype=tf.int32)

    # y = np.random.randint(0, height - dy + 1)
    y = g1.uniform(shape=(), minval=0, maxval=height - dy + 1, dtype=tf.int32)

    return img[y:(y + dy), x:(x + dx), :], (float(x) / img.shape[1]) * img_real_size, (
                float(y) / img.shape[0]) * img_real_size


# I assume that this is a map for a SINGLE image, it does not work on batches of images
# Vectorizing the map (that is, making it operate on batches) will not achieve any appreciable speedup unless the random crop operation (see random_crop_single) is also vectorized
# e.g. the randomness is on a per batch rather than an individual image level
# But I think that is not enough randomization
def image_crop_map(image, labels):
    image_crop, xshift, yshift = random_crop_single(image, (224, 224), 3000)

    shiftarray = tf.stack([tf.cast(xshift, tf.float32), tf.cast(yshift, tf.float32)], axis=0)
    shiftarray = tf.reshape(shiftarray, [1, 2])
    # shiftarray = tf.transpose(shiftarray)

    # tf.print("original", shiftarray, output_stream=sys.stderr)
    # We must scale the shifts to match the scale of the labels
    # shiftarray[:, 0] *= scalexparam
    # The scaling is inverted in y direction since north is positive in the coordinate system but going up is negative
    # shiftarray[:, 1] *= -scaleyparam

    shiftarray = tf.math.multiply(
        [scalexparam, scaleyparam], shiftarray, name=None
    )

    # tf.print("scaled", shiftarray, output_stream=sys.stderr)

    returnlabels = tf.add(labels, shiftarray)

    # tf.print("summed", returnlabels, output_stream=sys.stderr)
    # Code you would use if the base labels were not already scaled: we save an addition operation doing the scaling earlier
    # returnlabels = tf.math.multiply(returnlabels, scaler.scale_)
    # returnlabels = tf.math.add(returnlabels, scaler.min_)

    # TF actually automatically does the reshaping if we don't have this line, but I added it to make sure
    returnlabels = tf.reshape(returnlabels, [2, ])

    return image_crop, returnlabels


# It was necessary to wrap my map code in tf.py_function which doesn't accept boolean arguments
# I'm sure there's some combination of lambdas or chain calls we can do to continue to use our is_valid flag properly
# But to just get things working fast, I simply copied my above functions without the random part for the validation set
def random_crop_single_valid(img, random_crop_size, img_real_size):
    # Note: image_data_format is 'channel_last'
    # print("IMAGE SHAPE" , img.shape)
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]

    dy, dx = random_crop_size

    x = 0
    y = 0
    # Second and third return values are the x and y adjustments for labels (unscaled)
    return img[y:(y + dy), x:(x + dx), :], (float(x) / img.shape[1]) * img_real_size, (
                float(y) / img.shape[0]) * img_real_size


# I assume that this is a map for a SINGLE image, it does not work on batches of images
# This is easy to change though I don't think it will speed things up
def image_crop_map_valid(image, labels):
    image_crop, xshift, yshift = random_crop_single_valid(image, (224, 224), 3000)

    # print(shiftarray)

    # tf.print("shift", shiftarray, "labels", labels, "returnlabels", returnlabels, output_stream=sys.stderr)
    # returnlabels = labels

    # Code you would use if the base labels were not already scaled: we save an addition operation doing the scaling earlier
    # returnlabels = tf.math.multiply(returnlabels, scaler.scale_)
    # returnlabels = tf.math.add(returnlabels, scaler.min_)

    # TF actually automatically does the reshaping if we don't have this line, but I added it to make sure
    returnlabels = tf.reshape(labels, [2, ])

    return image_crop, returnlabels


AUTOTUNE = tf.data.AUTOTUNE
# Batch_size from the generator - we unbatch immediately but this determines how many things are loaded at once by the generator
gen_batch_size = 1
# Batch size for the actual dataset
dataset_batch = 128
# Size, in pixels, of final cropped input image
input_size = 224;
# real world size of image pre cropping
imageSizeInMeters = 3000;
# Rescaling operation for pixel values
datagen = ImageDataGenerator(rescale=1. / 255)
num_classes = 2
# Training example generator, grabs images based on filenames in dataframe, as well as labels from dataframe
# Target size is size, in pixels, that images are resized to BEFORE cropping
train_generator = datagen.flow_from_dataframe(dataframe=train, directory=traindatapath, x_col="Filename",
                                              y_col=["x", "y"], class_mode="other", target_size=(224, 224),
                                              batch_size=gen_batch_size, shuffle=False)



validgen = ImageDataGenerator(rescale=1. / 255)
valid_generator = validgen.flow_from_dataframe(dataframe=test, directory=testdatapath, x_col="Filename",
                                               y_col=["x", "y"], class_mode="other", target_size=(224, 224),
                                               batch_size=gen_batch_size, shuffle=False)

train_ds = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, input_size, input_size, 3], [None, 2])
)

valid_ds = tf.data.Dataset.from_generator(
    lambda: valid_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, input_size, input_size, 3], [None, 2])
)

# This takes all samples possible from the generator - it is a complete dataset so we don't need repeat
train_ds = train_ds.take(train_generator.n)

# unbatching because our image crop function expects single images
train_ds = train_ds.unbatch()

# Cache, also saves the cache into a file called traincache.  This file is SHARED between different runs which have this line!
# Caching means that all of the above operations (image loading and initial resizing) should only ever need to be run ONCE
# Cache must be deleted if data set, or any prior operation, is altered
train_ds = train_ds.cache(str(workingpath / "traincache"))

# Shuffle the data, buffer size of 500 is arbitrary. A true random shuffle of 24,000ish samples is excessive
train_ds = train_ds.shuffle(256, reshuffle_each_iteration=True)

# Image crop, batch, and setting prefetch
train_ds = train_ds.map(image_crop_map, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).batch(
    dataset_batch).prefetch(buffer_size=AUTOTUNE)


# Repeat of above operations for validation set
valid_ds = valid_ds.take(valid_generator.n)
valid_ds = valid_ds.unbatch()
valid_ds = valid_ds.cache(str(workingpath /  "testcache"))

# Note the is_valid=true flag in the lambda function to pass to crop function
valid_ds = valid_ds.map(image_crop_map_valid, num_parallel_calls=tf.data.AUTOTUNE).batch(dataset_batch).prefetch(
    buffer_size=AUTOTUNE)

# Xception base model, include_top = false and pooling=None means that we must manually implement final layers after convolutional layers (which we do below)
# We do this because we want to edit them
# Classifier_activation doesn't do anything since we have include_top=false
xceptionmodel = Xception(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classifier_activation="linear",
)

# regularization parameters for l1 and l2 regs set from args from command line
penalty_l2 = l2_reg
penalty_l1 = l1_reg

regularizer = tf.keras.regularizers.l1_l2(penalty_l1, penalty_l2)

# This convoluted means of setting regularization is/was necessary because of a bug in TF (google the code snippet)
# May no longer be necessary but this does work
for layer in xceptionmodel.layers:
    for attr in ['kernel_regularizer']:
        if hasattr(layer, attr):
            setattr(layer, attr, regularizer)
model_json = xceptionmodel.to_json()
xceptionmodel = tf.keras.models.model_from_json(model_json)

# Our loss function
neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

# These step sizes are used to set epoch sizes when we are using the generators instead of finite datasets
STEP_SIZE_TRAIN = train_generator.n // dataset_batch
STEP_SIZE_VALID = valid_generator.n // dataset_batch

# just some name shortening
tfd = tfp.distributions

# This is a functional implementation of the final (top) layers of our xception model
x = tfk.layers.GlobalAveragePooling2D(name='avg_pool')(xceptionmodel.output)
x = tfk.layers.Dropout(.25)(x)
x = tfk.layers.Dense(4096, activation='relu', kernel_regularizer=tfk.regularizers.l1_l2(penalty_l1, penalty_l2))(x)
x = tfk.layers.Dropout(.25)(x)
x = tfk.layers.Dense(2048, activation='relu', kernel_regularizer=tfk.regularizers.l1_l2(penalty_l1, penalty_l2))(x)
x = tfk.layers.Dropout(.25)(x)
x = tfk.layers.Dense(1024, activation='relu', kernel_regularizer=tfk.regularizers.l1_l2(penalty_l1, penalty_l2))(x)
x = tfk.layers.Dropout(.25)(x)
x = tfk.layers.Dense(1, activation='sigmoid', kernel_regularizer=tfk.regularizers.l1_l2(penalty_l1, penalty_l2))(x)
x = tfk.layers.Dropout(.25)(x)

# These layers in particular define the probabilistic distribution output layer
if distribution_type == 'normal':
    x = tfk.layers.Dense(tfp.layers.IndependentNormal.params_size(2))(x)
    finaldist = tfp.layers.IndependentNormal(2, convert_to_tensor_fn=tfd.Distribution.mean, name="output")(x)

elif distribution_type == 'lambertwnormal':
    x = tfk.layers.Dense(4)(x)
    finaldist = tfp.layers.DistributionLambda(lambda t:
                                              tfd.LambertWNormal(loc=t[..., :2],
                                                                 scale=1e-5 + tf.nn.softplus(.05 * t[..., 2:]),
                                                                 tailweight=.2), convert_to_tensor_fn=tfd.Distribution.mean)(x)
elif distribution_type == 'truncatednormal':
    x = tfk.layers.Dense(4)(x)
    finaldist = tfp.layers.DistributionLambda(lambda t:
                                              tfd.TruncatedNormal(loc=tf.clip_by_value(t[..., :2],0.,1.),
                                                                 scale=1e-5 + tf.nn.softplus(.05 * t[..., 2:]),
                                                                 low=-.1, high=1.1), convert_to_tensor_fn=tfd.Distribution.mean)(x)
elif distribution_type == 'truncatedcauchy':
    x = tfk.layers.Dense(4)(x)
    finaldist = tfp.layers.DistributionLambda(lambda t:
                                              tfd.TruncatedCauchy(loc=tf.clip_by_value(t[..., :2],0.,1.),
                                                                 scale=1e-5 + tf.nn.softplus(.05 * t[..., 2:]),
                                                                 low=-.1, high=1.1), convert_to_tensor_fn=tfd.Distribution.mean)(x)

elif distribution_type == 'laplace':
    x = tfk.layers.Dense(4)(x)
    finaldist = tfp.layers.DistributionLambda(lambda t:
                                              tfd.Laplace(loc=t[..., :2],
                                                                 scale=1e-5 + tf.nn.softplus(.05 * t[..., 2:]),
                                                                  ), convert_to_tensor_fn=tfd.Distribution.mean)(x)


elif distribution_type == 'mixture2':
    num_components = 2
    params_size = tfp.layers.MixtureSameFamily.params_size(
        num_components,
        component_params_size=tfp.layers.IndependentNormal.params_size(2))
    x = tfk.layers.Dense(params_size, activation=None)(x)
    finaldist = tfp.layers.MixtureSameFamily(num_components, tfp.layers.IndependentNormal(2), convert_to_tensor_fn=tfd.Distribution.mean)(x)
elif distribution_type == 'mixture3':
    num_components = 3
    params_size = tfp.layers.MixtureSameFamily.params_size(
        num_components,
        component_params_size=tfp.layers.IndependentNormal.params_size(2))
    x = tfk.layers.Dense(params_size, activation=None)(x)
    finaldist = tfp.layers.MixtureSameFamily(num_components, tfp.layers.IndependentNormal(2), convert_to_tensor_fn=tfd.Distribution.mean)(x)
else:
    print("Distribution type unknown, cancelling run")
    exit()

new_model = tf.keras.Model(inputs=xceptionmodel.input, outputs=finaldist)

# Model summary for troubleshooting
print(new_model.summary())


# ADAM optimizer
# Clip value is VERY important to have!
adamopt = tf.optimizers.Adam(learning_rate=initialLR, clipvalue=1)

# Model compile with metric and optimizer
new_model.compile(loss = 'categorical_crossentropy', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                                                      tf.keras.metrics.categorical_accuracy],
              optimizer=adamopt)

# Checkpoint naming structure
checkpoint_filepath = str(workingpath / f'{testname:s}' / (f'{testname:s}' + '_{epoch:02d}_{val_rmse:.5f}.hdf5'))

# Checkpoint for saving model when it improves on val_rmse
# Can be editing for different save behavior
checkpoint = ModelCheckpoint(checkpoint_filepath,
                             monitor='val_rmse',
                             verbose=2,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='max')


# Learning rate decay function
def step_decay(epoch):
    # initial_lrate = 6.06528e-05
    initial_lrate = initialLR
    drop = 0.95
    epochs_drop = 250.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


lrate = LearningRateScheduler(step_decay)

# End training run if NaN (otherwise it just continues with NaN forever)
terminateOnNaNCall = TerminateOnNaN()

# Another LR reduction schedule, for when learning plateaus (no improvement in 70 epochs for example)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_rmse', factor=0.7,
                                                 patience=70, min_lr=0.000000)

# Logging, saves to a log file
csv_logger = tf.keras.callbacks.CSVLogger(str(workingpath / f'{testname:s}' / f'{testname:1s}.log'), append=True)

callbacks_list = [lrate, reduce_lr, checkpoint, csv_logger, terminateOnNaNCall]



# Code for training when training using just generators
history = new_model.fit_generator(generator=train_generator,
                                  shuffle=True,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    max_queue_size=128,
                    epochs=epochs,
                    callbacks=callbacks_list,
                    verbose=2)



accuracy2 = new_model.evaluate(valid_generator)
print(accuracy2)

#Print out the performance in array form
print(f'Loss = {accuracy2[0]}')
print(f'RMSE = {accuracy2[1]}')
print(f'Categorical Accuracy = {accuracy2[2] * 100}%')