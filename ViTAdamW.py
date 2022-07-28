import tensorflow as tf
import numpy as np
import pandas as pd
import math
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import LearningRateScheduler
import pathlib
from PIL import Image
import tensorflow_probability as tfp
import sys
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import TerminateOnNaN


#No longer necessary but doesn't hurt
Image.MAX_IMAGE_PIXELS = 1000000000

#This puts the path of the current directory of the python file as the source
workingpath = pathlib.Path.cwd()

#These are the arguments given when running the file
#Keep in mind if you want to run this file in pycharm you need to set these flags in the run configuration
opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

#Name of the test, for saving checkpoints
testname = args[0]
#String used to determine the test dataset: must be a consistent and unique string at start of each test dataset image
testsetstring = args[1]
#Initial learning rate
initialLR = float(args[2])
#l1 regularization parameter on most layers
l1_reg = float(args[3])
#l2 regularization parameter on most layers
l2_reg = float(args[4])
#Training run length
epochs = int(args[5])

p = pathlib.Path(workingpath / f'{testname:s}')
p.mkdir(parents=True, exist_ok=True)



#This indicates that the data should be found one directory prior to the directory the file is run from
#Simply change to the correct directory if not
traindatapath = "/Users/santiagodorado/Documents/Project Images 2/ProjImages" #str(workingpath / '..' )
testdatapath = "/Users/santiagodorado/Documents/Project Images 2/ProjImages" #str(workingpath / '..' )
#Location of the labels file: the file should be a csv with  filename, x, and y columns
alldata = pd.read_csv('/Users/santiagodorado/Downloads/GeolocalizationZipped 4/Code/multiplesourcelabels.csv')


#Divides out the test dataset from the train dataset
testmatch = [x.startswith(testsetstring) for x in alldata['Filename'].values]
trainmatch = [not i for i in testmatch]
train = alldata[trainmatch].copy()
test = alldata[testmatch].copy()

#Coordinates of RoI from ArcGIS, the plus 1500 meters due to the fact that the coords are upper left corners of 3k chips (but arcgis samples as if from center)
topRoi = 4005539.977963 + 1500
bottomRoi = 3971320.265500 + 1500
leftRoi = 380030.074942 - 1500
rightRoi = 414249.787405 - 1500

heightRoi = topRoi - bottomRoi
widthRoi = rightRoi - leftRoi

centerY = (topRoi + bottomRoi) / 2.0
centerX = (rightRoi + leftRoi) / 2.0

sideWidth = 1.0

#Bounds for size restriction experiment, training set
topBound = centerY + sideWidth/2.0 * heightRoi
bottomBound = centerY - sideWidth/2.0 * heightRoi
leftBound = centerX - sideWidth/2.0 * widthRoi
rightBound = centerX + sideWidth/2.0 * widthRoi

#Train set restrictions
inareax = [((x > (leftBound)) and (x< (rightBound) )) for x in train['x'].values]
train = train[inareax]
inareay = [((x > (bottomBound)) and (x< (topBound ) )) for x in train['y'].values]
train = train[inareay]

#Validation padding to avoid boundary effects
validPad = 1500

#Additional area restriction for test set only
inareax = [((x > (leftBound + validPad)) and (x< (rightBound - validPad) )) for x in test['x'].values]
test = test[inareax]
inareay = [((x > (bottomBound + validPad)) and (x< (topBound - validPad) )) for x in test['y'].values]
test = test[inareay]


#experimental area restrictions

#Additionally removed the ADOP2001 dataset from most runs due to it being false color infrared
remove = [x.startswith("CHIP_ADOP2001") for x in train['Filename'].values]
notremove = [not i for i in remove]
train =  train[notremove].copy()


#This scaler will be used to scale the coordinates to between 0 and 1
#Scaling is based on ALL data, and is applied equally to train and test
#Now it is very important to make sure that, when we adjust labels during cropping, we don't screw things up with scaling
#Thus, we need to either combine the labels first THEN scale, or make sure to save just the scaling portion to scale the adjustment (current approach)
scaler = MinMaxScaler(feature_range=(0, 1))

alldata[['x', 'y']] = scaler.fit_transform(alldata[['x', 'y']])

#Scale parameters for x and y columns, for later use
scalexparam = scaler.scale_[0]
scaleyparam = scaler.scale_[1]

train[['x', 'y']] = scaler.transform(train[['x', 'y']])
#print(test.head(5))
test[['x', 'y']] = scaler.transform(test[['x', 'y']])
#print(test.head(5))


#Replacement for random crop above: note that ROWS are our y coordinate, so y coordinate is FIRST in image matrix
#If we are making validation data, we always crop from the upper left (no adjustment of label), so there is no randomness
#Flag is currently not used (see validation versions of these functions below) due to not fully refactoring code

def random_crop_single(img, random_crop_size, img_real_size, is_valid = False):
    # Note: image_data_format is 'channel_last'
    #print("IMAGE SHAPE" , img.shape)
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]

    dy, dx = random_crop_size
    #Choose a random point in the valid area to be the new location, then crop
    if not is_valid:
        x = np.random.randint(0, width - dx + 1)
        #y = tf.experimental.numpy.random.randint(0 , height - dy + 1, dtype=np.int)
        #tf.print("y", int(y), output_stream=sys.stderr)
        y = np.random.randint(0, height - dy + 1)
        #y = int(y)

    else:
        x = 0
        y = 0
    #Second and third return values are the x and y adjustments for labels (unscaled)
    return img[ y:(y+dy), x:(x+dx), :], (float(x) / img.shape[1]) * img_real_size, (float(y) / img.shape[0]) * img_real_size



#I assume that this is a map for a SINGLE image, it does not work on batches of images
#Vectorizing the map (that is, making it operate on batches) will not achieve any appreciable speedup unless the random crop operation (see random_crop_single) is also vectorized
#e.g. the randomness is on a per batch rather than an individual image level
#But I think that is not enough randomization
def image_crop_map(image, labels, is_valid=False):
    image_crop, xshift, yshift = random_crop_single(image, (224,224), 3000, is_valid)

    #Combine shifts into one numpy array
    shiftarray = np.column_stack([xshift, yshift])

    #We must scale the shifts to match the scale of the labels
    shiftarray[:, 0] *= scalexparam
    #The scaling is inverted in y direction since north is positive in the coordinate system but going up is negative
    shiftarray[:, 1] *= -scaleyparam

    #print(shiftarray)


    returnlabels = tf.add(labels, shiftarray)

    #This line is absolutely essential because apparently if you don't have it, because the shape is [1,2], tensorflow probability assigns loss batch_num times per item in batch
    #So each batch has batch^2 number of losses...
    #I can't see this as anything other than a bug but anyway, this line avoids it
    returnlabels = tf.reshape(returnlabels, [2, ])

    return image_crop, returnlabels


#I assume that this is a map for a SINGLE image, it does not work on batches of images
#This is easy to change though I don't think it will speed things up
def image_crop_map_valid(image, labels, is_valid=False):
    image_crop, xshift, yshift = random_crop_single_valid(image, (224,224), 3000, is_valid)
    #TF actually automatically does the reshaping if we don't have this line, but I added it to make sure
    returnlabels = tf.reshape(labels, [2, ])

    return image_crop, returnlabels


AUTOTUNE = tf.data.AUTOTUNE
#Batch_size from the generator - we unbatch immediately but this determines how many things are loaded at once by the generator
gen_batch_size = 1
#Batch size for the actual dataset
dataset_batch=128
#Size, in pixels, of final cropped input image
input_size = 224;
#real world size of image pre cropping
imageSizeInMeters = 3000;
#Rescaling operation for pixel values
datagen=ImageDataGenerator(rescale=1./255)
num_classes = 2


#Training example generator, grabs images based on filenames in dataframe, as well as labels from dataframe
#Target size is size, in pixels, that images are resized to BEFORE cropping
train_generator=datagen.flow_from_dataframe(dataframe=train,
                                            directory=traindatapath,
                                            x_col="Filename", y_col=["x","y"],
                                            class_mode="other",
                                            target_size=(224,224), #target_size=(448,448),
                                            batch_size=gen_batch_size, shuffle=False)


validgen=ImageDataGenerator(rescale=1./255)
valid_generator=validgen.flow_from_dataframe(dataframe=test,
                                             directory=testdatapath,
                                             x_col="Filename", y_col=["x","y"], class_mode="other",
                                             target_size=(224,224), #target_size=(448,448),
                                             batch_size=gen_batch_size, shuffle=False)

#from_generator is still subject to the python GIL, and thus is slower (in theory) than a pure TF implementation
#since I cache after loading the images, this is of minor importance
#However, I still intend to switch over to a pure tf image loading scheme soon
train_ds = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None,input_size,input_size,3], [None,2])
)


valid_ds = tf.data.Dataset.from_generator(
    lambda: valid_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None,input_size,input_size,3], [None,2])
)

#This takes all samples possible from the generator - it is a complete dataset so we don't need repeat
train_ds = train_ds.take(train_generator.n)


#unbatching because our image crop function expects single images
train_ds = train_ds.unbatch()
#train_ds=train_ds.repeat()



#Shuffle the data, buffer size of 500 is arbitrary. A true random shuffle of 24,000ish samples is excessive
train_ds = train_ds.shuffle(160, reshuffle_each_iteration=True)

#Image crop, batch, and setting prefetch
train_ds = train_ds.map(lambda x, y: tf.py_function(image_crop_map, [x,y],[tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).batch(dataset_batch).prefetch(buffer_size=AUTOTUNE)


#Repeat of above operations for validation set
valid_ds = valid_ds.take(valid_generator.n)
valid_ds = valid_ds.unbatch()

#Note the is_valid=true flag in the lambda function to pass to crop function
valid_ds = valid_ds.map(lambda x, y: tf.py_function(image_crop_map_valid, [x,y],[tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE,deterministic=False).batch(dataset_batch).prefetch(buffer_size=AUTOTUNE)

from ArchitectureExpExample.vit2_model import VisionTransformer

vitmodel = VisionTransformer(
    image_size=224,
    patch_size=16,
    num_layers=4,#scaled down since it had no big impact on improving the rmse or accuracy, plus it would run 2x faster
    num_classes=tfp.layers.IndependentNormal.params_size(2),
    d_model=1024,
    num_heads=16,
    mlp_dim=2048,
    channels=3,
    dropout=0.75

)
input = tf.keras.layers.Input((224,224,3))

vitmodel = vitmodel(input)

#regularization parameters for l1 and l2 regs set from args from command line
penalty_l2 = l2_reg
penalty_l1 = l1_reg

regularizer = tf.keras.regularizers.l1_l2(penalty_l1,penalty_l2)

#These step sizes are used to set epoch sizes when we are using the generators instead of finite datasets
STEP_SIZE_TRAIN=train_generator.n//dataset_batch
STEP_SIZE_VALID=valid_generator.n//dataset_batch

#just some name shortening
tfd = tfp.distributions

#This is a functional implementation of the final (top) layers of our xception model
finaldist = tfp.layers.IndependentNormal(2,convert_to_tensor_fn=tfd.Distribution.mean, name="output")(vitmodel)

#Combined the inputs and outputs
pre_model = tf.keras.Model(inputs = input, outputs = finaldist)

#Layer Manipulation on the ViT Model
x = tf.keras.layers.Flatten()(pre_model.output)
x = tf.keras.layers.Dense(2048, activation='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(16, activation='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(8, activation='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(4, activation='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(2, activation='relu')(x)
x = tf.keras.layers.Dropout(0.25)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
x = tf.keras.layers.Dropout(0.25)(x)

#Combined the premodel inputs and the layer outputs
new_model = tf.keras.Model(pre_model.inputs, x)
#Model summary for troubleshooting
print(new_model.summary())


#ADAM optimizer
#Clip value is VERY important to have!
import tensorflow_addons as tfa

#A bug occured when trying to implement this learning rate. It said that object has no attribute dtype
# cyclical_learning_rate = tfa.optimizers.CyclicalLearningRate(
#     initial_learning_rate=initialLR,
#     maximal_learning_rate=2E-1,
#     step_size=376,
#     scale_fn=lambda y: (1 / (1.01 ** (y-1))),
#     scale_mode='cycle')

#AdamW optimizer that includes the initial learning rate
adamopt = tfa.optimizers.AdamW(learning_rate=initialLR, weight_decay=.00001, clipvalue=1)

#Model compile with metric and optimizer
new_model.compile(loss='categorical_crossentropy', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                                                      tf.keras.metrics.categorical_accuracy],
              optimizer=adamopt)


#Checkpoint naming structure
checkpoint_filepath = str(workingpath / f'{testname:s}'/  (f'{testname:s}' + '_{epoch:02d}_{val_rmse:.5f}.hdf5'))

#Checkpoint for saving model when it improves on val_rmse
#Can be editing for different save behavior
checkpoint = ModelCheckpoint(checkpoint_filepath,
                            monitor='val_rmse',
                            verbose=2,
                            save_best_only=True,
                            save_weights_only=False,
                            mode='min')

#Learning rate decay function
def step_decay(epoch):
    # initial_lrate = 6.06528e-05
    initial_lrate = initialLR
    drop = 0.0
    epochs_drop = 250.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)

#End training run if NaN (otherwise it just continues with NaN forever)
terminateOnNaNCall = TerminateOnNaN()

#Another LR reduction schedule, for when learning plateaus (no improvement in 70 epochs for example)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_rmse', factor=0.7,
                              patience=5, min_lr=0.000000)

#Logging, saves to a log file
csv_logger = tf.keras.callbacks.CSVLogger(str(workingpath / f'{testname:s}'/ f'{testname:1s}.log'), append=True)

#Combines all the callbacks
callbacks_list = [lrate, reduce_lr, checkpoint, csv_logger, terminateOnNaNCall]

#Code for training when training using just generators
history = new_model.fit_generator(generator=train_generator,
                                  shuffle=True,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    max_queue_size=128,
                    epochs=epochs,
                    callbacks=callbacks_list,
                    verbose=2)

#Evaluate the model on the withheld data
accuracy1 = new_model.evaluate(valid_generator)

#Print out the performance in array form
print(f'Loss = {accuracy1[0]}')
print(f'RMSE = {accuracy1[1]}')
print(f'Categorical Accuracy = {accuracy1[2] * 100}%')