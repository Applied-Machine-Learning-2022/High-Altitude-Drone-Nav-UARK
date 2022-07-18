import tensorflow
import tensorflow as tf
import tensorflow.python.keras as tfk
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
#from vit_model import ViT
from tensorflow.python.keras.callbacks import TerminateOnNaN
from pathlib import Path


#Previously when I had very big images this was needed to prevent a crash, no longer necessary but doesn't hurt
Image.MAX_IMAGE_PIXELS = 1000000000

#This can enhance speed by decreasing floating point precision, optional
#There are some additional tweaks you may need to do if you go this route, look it up on google
#keras.backend.set_floatx('float16')

#tf.config.run_functions_eagerly(True)
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
traindatapath = str(workingpath / '..' )
testdatapath = str(workingpath / '..' )
#Location of the labels file: the file should be a csv with  filename, x, and y columns
alldata = pd.read_csv('../multiplesourcelabels.csv')




#Area restriction code, for filtering data into only images from a smaller area if desired
#Values need to be determined in ArcGIS
# inareax = [((x > 378865) and (x<411552) ) for x in alldata['x'].values]
# alldata = alldata[inareax]
# inareay = [((x > 3971722) and (x<4004690) ) for x in alldata['y'].values]
# alldata = alldata[inareay]

#Divides out the test dataset from the train dataset
testmatch = [x.startswith(testsetstring) for x in alldata['Filename'].values]
trainmatch = [not i for i in testmatch]
train = alldata[trainmatch].copy()
test = alldata[testmatch].copy()
# test = alldata[trainmatch].copy()

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

#I additionally remove the ADOP2001 dataset from most runs due to it being false color infrared
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

# print("scalex", 500* scalexparam)
# print(scaleyparam)

train[['x', 'y']] = scaler.transform(train[['x', 'y']])
#print(test.head(5))
test[['x', 'y']] = scaler.transform(test[['x', 'y']])
#print(test.head(5))

import sys

#Seed for numpy if wanted
#np.random.seed(seed=2)

#When we do random crops, we must change the values, but in order to do so we must know the scaling we used
#Obsolete scaling code, from when I thought I needed to have seperate scalers for x and y
def makeScalers(dataframe):
    xscaledata = []
    yscaledata = []
    for index, row in dataframe.iterrows():
        xscaledata.append(row['x'])
        yscaledata.append(row['y'])

    seriesscalex = Series(xscaledata)
    seriesscaley = Series(yscaledata)

    valuesscalex = seriesscalex.values
    valuesscalex = valuesscalex.reshape((len(valuesscalex), 1))

    valuesscaley = seriesscaley.values
    valuesscaley = valuesscaley.reshape((len(valuesscaley), 1))

    # Actually we want to use the SAME scaling on test data of course!

    scalerx = MinMaxScaler(feature_range=(0, 1))
    scalerx = scalerx.fit(valuesscalex)

    scalery = MinMaxScaler(feature_range=(0, 1))
    scalery = scalery.fit(valuesscaley)

    return scalerx, scalery

#More obsolete scaling code
def makeTransform(dataframe, scalerx, scalery):
    xdata = []
    ydata = []
    for index, row in dataframe.iterrows():
        xdata.append(row['x'])
        ydata.append(row['y'])

    seriesscalex = Series(xdata)
    seriesscaley = Series(ydata)

    valuesscalex = seriesscalex.values
    valuesscalex = valuesscalex.reshape((len(valuesscalex), 1))

    valuesscaley = seriesscaley.values
    valuesscaley = valuesscaley.reshape((len(valuesscaley), 1))

    # Actually we want to use the SAME scaling on test data of course!
    x = scalerx.transform(valuesscalex)
    y = scalery.transform(valuesscaley)

    return x, y

scalerx, scalery = makeScalers(alldata)

#In case you want the scaled labels of the test set in a file
# testx, testy = makeTransform(test, scalerx, scalery)
# numpy.savetxt("foo.csv", np.hstack((testx,testy)), delimiter=",")

#Old random crop code for generator based data input
def random_crop(img, random_crop_size, img_real_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :], (float(x) / img.shape[1]) * img_real_size, (float(y) / img.shape[0]) * img_real_size

#Generates crops, now replaced by map function below
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

            #print(batch_y[i][0])
            #print (xarray[0])
            xarray[0] += xshift * scalexparam
            yarray[0] -= yshift * scaleyparam
            #xarray.reshape(1,-1)
            xarray = [[xarray[0]]]
            yarray = [[yarray[0]]]
            #print(xarray)
            #print(scalerx.transform(xarray))
            #print("TRANSFORMED", scalerx.transform(xarray)[0][0])
            batch_y[i][0], batch_y[i][1]  = xarray[0][0], yarray[0][0]
            # print(batch_y[i])

        yield (batch_crops, batch_y)



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
    #print(shiftarray)
    #tf.print("shift", shiftarray, output_stream=sys.stderr)
    #We must scale the shifts to match the scale of the labels
    shiftarray[:, 0] *= scalexparam
    #The scaling is inverted in y direction since north is positive in the coordinate system but going up is negative
    shiftarray[:, 1] *= -scaleyparam

    #print(shiftarray)


    returnlabels = tf.add(labels, shiftarray)
    #tf.print("shift", shiftarray, "labels", labels, "returnlabels", returnlabels, output_stream=sys.stderr)
    #returnlabels = labels

    #Code you would use if the base labels were not already scaled: we save an addition operation doing the scaling earlier
    # returnlabels = tf.math.multiply(returnlabels, scaler.scale_)
    # returnlabels = tf.math.add(returnlabels, scaler.min_)

    #This line is absolutely essential because apparently if you don't have it, because the shape is [1,2], tensorflow probability assigns loss batch_num times per item in batch
    #So each batch has batch^2 number of losses...
    #I can't see this as anything other than a bug but anyway, this line avoids it
    returnlabels = tf.reshape(returnlabels, [2, ])

    return image_crop, returnlabels

#It was necessary to wrap my map code in tf.py_function which doesn't accept boolean arguments
#I'm sure there's some combination of lambdas or chain calls we can do to continue to use our is_valid flag properly
#But to just get things working fast, I simply copied my above functions without the random part for the validation set
def random_crop_single_valid(img, random_crop_size, img_real_size, is_valid = False):
    # Note: image_data_format is 'channel_last'
    #print("IMAGE SHAPE" , img.shape)
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]

    dy, dx = random_crop_size

    x = 0
    y = 0
    #Second and third return values are the x and y adjustments for labels (unscaled)
    return img[ y:(y+dy), x:(x+dx), :], (float(x) / img.shape[1]) * img_real_size, (float(y) / img.shape[0]) * img_real_size



#I assume that this is a map for a SINGLE image, it does not work on batches of images
#This is easy to change though I don't think it will speed things up
def image_crop_map_valid(image, labels, is_valid=False):
    image_crop, xshift, yshift = random_crop_single_valid(image, (224,224), 3000, is_valid)



    #print(shiftarray)



    #tf.print("shift", shiftarray, "labels", labels, "returnlabels", returnlabels, output_stream=sys.stderr)
    #returnlabels = labels

    #Code you would use if the base labels were not already scaled: we save an addition operation doing the scaling earlier
    # returnlabels = tf.math.multiply(returnlabels, scaler.scale_)
    # returnlabels = tf.math.add(returnlabels, scaler.min_)

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
train_generator=datagen.flow_from_dataframe(dataframe=train, directory=traindatapath, x_col="Filename", y_col=["x","y"], class_mode="other", target_size=(448,448), batch_size=gen_batch_size, shuffle=False)

#Old code when entire input pipeline was generator based
#train_crops = crop_generator(train_generator, input_size, imageSizeInMeters, scalerx, scalery)
#print("NEXT", next(train_crops)[1])

validgen=ImageDataGenerator(rescale=1./255)
valid_generator=validgen.flow_from_dataframe(dataframe=test, directory=testdatapath, x_col="Filename", y_col=["x","y"], class_mode="other",target_size=(448,448), batch_size=gen_batch_size, shuffle=False)

#valid_crops = crop_generator(valid_generator, input_size, imageSizeInMeters, scalerx, scalery)


#from_generator is still subject to the python GIL, and thus is slower (in theory) than a pure TF implementation
#since I cache after loading the images, this is of minor importance
#However, I still intend to switch over to a pure tf image loading scheme soon
train_ds = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None,input_size*2,input_size*2,3], [None,2])
)


valid_ds = tf.data.Dataset.from_generator(
    lambda: valid_generator,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None,input_size*2,input_size*2,3], [None,2])
)

#This takes all samples possible from the generator - it is a complete dataset so we don't need repeat
train_ds = train_ds.take(train_generator.n)


#unbatching because our image crop function expects single images
train_ds = train_ds.unbatch()
#train_ds=train_ds.repeat()

#I tried to make a vectorized map but it is actually not possible unless we would accept every crop operation on each image being the same (e.g. same crop and label adjustment)
#This may actually be OK to do, crops would still differ between batches
#Note that the map function must be rewritten to be vectorized
#vect_map = tf.vectorized_map(single_image_crop_map, batch_size)

#Cache, also saves the cache into a file called traincache.  This file is SHARED between different runs which have this line!
#Caching means that all of the above operations (image loading and initial resizing) should only ever need to be run ONCE
#Cache must be deleted if data set, or any prior operation, is altered
train_ds = train_ds.cache(str(workingpath / f'{testname:s}' / "traincache"))



#Shuffle the data, buffer size of 500 is arbitrary. A true random shuffle of 24,000ish samples is excessive
train_ds = train_ds.shuffle(160, reshuffle_each_iteration=True)

#Image crop, batch, and setting prefetch
train_ds = train_ds.map(lambda x, y: tf.py_function(image_crop_map, [x,y],[tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).batch(dataset_batch).prefetch(buffer_size=AUTOTUNE)


# #list(dataset.as_numpy_iterator())
# i=0
# for element in train_ds:
#     print("ELEMENT", element[1])
#     i += 1
#     if i > 1:
#         break


#sys.exit("Error message")

#Repeat of above operations for validation set
valid_ds = valid_ds.take(valid_generator.n)
valid_ds = valid_ds.unbatch()
valid_ds = valid_ds.cache(str(workingpath / f'{testname:s}' / "testcache"))

#Note the is_valid=true flag in the lambda function to pass to crop function
valid_ds = valid_ds.map(lambda x, y: tf.py_function(image_crop_map_valid, [x,y],[tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE,deterministic=False).batch(dataset_batch).prefetch(buffer_size=AUTOTUNE)

#Xception base model, include_top = false and pooling=None means that we must manually implement final layers after convolutional layers (which we do below)
#We do this because we want to edit them
#Classifier_activation doesn't do anything since we have include_top=false



from vit2_model_regularized import VisionTransformer

vitmodel = VisionTransformer(
    image_size=224,
    patch_size=28,
    num_layers=6,
    num_classes=tfp.layers.IndependentNormal.params_size(2),
    d_model=1024,
    num_heads=16,
    mlp_dim=2048,
    channels=5,
    dropout=0.3,

)
from coord import CoordinateChannel2D
input = tf.keras.layers.Input((224,224,3))
coorded = CoordinateChannel2D()(input)

#vitmodel.build((None,224,224,3))

vitmodel = vitmodel(coorded)

#regularization parameters for l1 and l2 regs set from args from command line
penalty_l2 = l2_reg
penalty_l1 = l1_reg

regularizer = tf.keras.regularizers.l1_l2(penalty_l1,penalty_l2)

#This convoluted means of setting regularization is/was necessary because of a bug in TF (google the code snippet)
#May no longer be necessary but this does work
# for layer in vitmodel.layers:
#     for attr in ['kernel_regularizer']:
#         if hasattr(layer, attr):
#           setattr(layer, attr, regularizer)
# model_json = vitmodel.to_json()
# vitmodel = tf.keras.models.model_from_json(model_json)

#Our loss function
neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

#These step sizes are used to set epoch sizes when we are using the generators instead of finite datasets
STEP_SIZE_TRAIN=train_generator.n//dataset_batch
STEP_SIZE_VALID=valid_generator.n//dataset_batch

#just some name shortening
tfd = tfp.distributions

from tensorflow.python.keras import layers



#This is a functional implementation of the final (top) layers of our xception model

finaldist = tfp.layers.IndependentNormal(2,convert_to_tensor_fn=tfd.Distribution.mean, name="output")(vitmodel)


new_model = tf.keras.Model(inputs = input, outputs = finaldist)
#print(new_model.summary)

#sgd, if desired to be used instead of ADAM optimizer
sgd = SGD(lr=0.0005, decay=0.0, momentum=0.9, nesterov=True, clipvalue = 1)





#Some plotting code
# from keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)

#Model summary for troubleshooting
print(new_model.summary())

#Weight loading from a prior run if desired
#model.load_weights("C:\\Users\\Winthrop\\Desktop\\checkpoints\\NOTHING.hdf5")

#ADAM optimizer
#Clip value is VERY important to have!
adamopt = tf.optimizers.Adam(learning_rate=.0001, clipvalue=1)

#Model compile with metric and optimizer
new_model.compile(loss = neg_log_likelihood, metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')],
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
    drop = 0.95
    epochs_drop = 250.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay)

#End training run if NaN (otherwise it just continues with NaN forever)
terminateOnNaNCall = TerminateOnNaN()

#Another LR reduction schedule, for when learning plateaus (no improvement in 70 epochs for example)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_rmse', factor=0.7,
                              patience=70, min_lr=0.000000)

#Logging, saves to a log file
csv_logger = tf.keras.callbacks.CSVLogger(str(workingpath / f'{testname:s}'/ f'{testname:1s}.log'), append=True)

#Tensorboard callback, if desired
#tensorboardcb = keras.callbacks.TensorBoard(log_dir='.\\logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')


callbacks_list = [ checkpoint, csv_logger, terminateOnNaNCall]

#callbacks_list = [checkpoint]


#Runs the training
history = new_model.fit(train_ds, shuffle=True, validation_data=valid_ds,
                    max_queue_size=16,
                    epochs=epochs, callbacks=callbacks_list,verbose=2)

#Obsolete code for training when training using just generators
# history = new_model.fit_generator(generator=train_crops, shuffle=True,
#                     steps_per_epoch=STEP_SIZE_TRAIN,
#                     validation_data=valid_crops,
#                     validation_steps=STEP_SIZE_VALID, max_queue_size=128,
#                     epochs=epochs, callbacks=callbacks_list,verbose=2)



#Saves the history (Note, only saves if run is completed successfully
with open(str(workingpath.joinpath(f'{testname:s}', f'historysave{testname}.pkl')), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

#This chunk of code will search all the checkpoints in the folder, then generate and save predictions, stddev, etc. for the BEST checkpoint
#Naturally, it assumes that ONLY one runs worth of checkpoints is in the target folder and that the naming conventions for filesaving above are used
savepath = workingpath.joinpath(f'{testname:s}')
filenames = savepath.glob("*.hdf5")
filelist = []
numbers = []
for file in filenames:
    filelist.append(str(file))
    file = file.with_suffix('')
    file = str(file)
    l = []
    for t in file.split("_"):
        try:
            l.append(float(t))
        except ValueError:
            pass
    try:
        numbers.append(l[-1])
    #If for whatever reason we don't find a number in a checkpoint file, we just assign it a very large number so it isn't chosen
    #but we want the number in the list to maintain index parity between this list and the filename list
    #that is, filename[i] should correspond to numbers[i] always
    except IndexError:
        numbers.append(100000)
        pass

print(numbers)

val, idx = min((val, idx) for (idx, val) in enumerate(numbers))

print(Path(filelist[idx]))


new_model.load_weights(str(Path(filelist[idx])))

trues = []
means = []
stddevs = []

for element in train_ds.as_numpy_iterator():
    trues.append(element[1])
    dist = new_model(element[0])
    means.append(dist.mean().numpy())
    stddevs.append(dist.stddev().numpy())



with open(str(workingpath.joinpath(f'{testname:s}', f'trues{testname}.pkl')), 'wb') as file_pi:
    pickle.dump(trues, file_pi)

with open(str(workingpath.joinpath(f'{testname:s}', f'means{testname}.pkl')), 'wb') as file_pi:
    pickle.dump(means, file_pi)

with open(str(workingpath.joinpath(f'{testname:s}', f'stddevs{testname}.pkl')), 'wb') as file_pi:
    pickle.dump(stddevs, file_pi)

#Obsolete method of saving predictions using generators
# valid_generator.reset()
# pred=model.predict_generator(valid_generator,
# steps=STEP_SIZE_VALID,
# verbose=2)
#
# print (pred)
#
# with open('C:\\Users\\Winthrop\\Desktop\\checkpoints\\NOTHING.pkl', 'wb') as file_pi:
#     pickle.dump(pred, file_pi)
