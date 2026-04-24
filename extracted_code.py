import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, 
                          Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D)
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Model, load_model, save_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

%matplotlib inline

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

from sklearn.model_selection import StratifiedKFold, cross_validate, LeaveOneGroupOut

from PIL import Image

def PlotClassFrequency(class_counts):
    plt.figure(figsize=(15,4))
    plt.bar(class_counts.index,class_counts)
    plt.xlabel('class')
    plt.xticks(np.arange(0, 10, 1.0))
    plt.ylabel('count')
    plt.title('Number of Images per Class')
    plt.show()

def DescribeImageData(data):
    print('Average number of images: ' + str(np.mean(data)))
    print("Lowest image count: {}. At: {}".format(data.min(), data.idxmin()))
    print("Highest image count: {}. At: {}".format(data.max(), data.idxmax()))
    print(data.describe())
    
def CreateImgArray(height, width, channel, data, folder, save_labels = True):
    """
    Writes image files found in 'imgs/train' to array of shape
    [examples, height, width, channel]
    
    Arguments:
    height -- integer, height in pixels
    width --  integer, width in pixels
    channel -- integer, number of channels (or dimensions) for image (3 for RGB)
    data -- dataframe, containing associated image properties, such as:
            subject -> string, alpha-numeric code of participant in image
            classname -> string, the class name i.e. 'c0', 'c1', etc. 
            img -> string, image name
    folder -- string, either 'test' or 'train' folder containing the images
    save_labels -- bool, True if labels should be saved, or False (just save 'X' images array).  
                   Note: only applies if using train folder
            
    Returns:
    .npy file -- file, contains the associated conversion of images to numerical values for processing
    """
    
    num_examples = len(data)
    X = np.zeros((num_examples,height,width,channel))
    if (folder == 'train') & (save_labels == True):
        Y = np.zeros(num_examples)
    
    for m in range(num_examples):
        current_img = data.img[m]
        img_path = 'imgs/' + folder + '/' + current_img
        img = image.load_img(img_path, target_size=(height, width))
        x = image.img_to_array(img)
        x = preprocess_input(x)
        X[m] = x
        if (folder == 'train') & (save_labels == True):
            Y[m] = data.loc[data['img'] == current_img, 'classname'].iloc[0]
        
    np.save('X_'+ folder + '_' + str(height) + '_' + str(width), X)
    if (folder == 'train') & (save_labels == True):
        np.save('Y_'+ folder + '_' + str(height) + '_' + str(width), Y)
        
def Rescale(X):
    return (1/(2*np.max(X))) * X + 0.5

def PrintImage(X_scaled, index, Y = None):
    plt.imshow(X_scaled[index])
    if Y is not None:
        if Y.shape[1] == 1:
            print ("y = " + str(np.squeeze(Y[index])))
        else:
            print("y = " + str(np.argmax(Y[index])))
            
def LOGO(X, Y, group, model_name, input_shape, classes, init, optimizer, metrics, epochs, batch_size):
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X, Y, group);
    cvscores = np.zeros((26,4))
    subject_id = []
    i = 0
    for train, test in logo.split(X, Y, group):
        # Create model
        model = model_name(input_shape = input_shape, classes = classes, init = init)
        # Compile the model
        model.compile(optimizer = optimizer, loss='sparse_categorical_crossentropy', metrics=[metrics])
        # Fit the model
        model.fit(X[train], Y[train], epochs = epochs, batch_size = batch_size, verbose = 0)
        # Evaluate the model
        scores_train = model.evaluate(X[train], Y[train], verbose = 0)
        scores_test = model.evaluate(X[test], Y[test], verbose = 0)
        # Save to cvscores
        cvscores[i] = [scores_train[0], scores_train[1] * 100, scores_test[0], scores_test[1] * 100]
        subject_id.append(group.iloc[test[0]])
        # Clear session
        K.clear_session()
        # Update counter
        i += 1
        
    return pd.DataFrame(cvscores, index = subject_id, columns=['Train_loss', 'Train_acc','Test_loss', 'Test_acc'])

driver_imgs_df = pd.read_csv('driver_imgs_list/driver_imgs_list.csv')
driver_imgs_df.head()

driver_imgs_df.shape

class_counts = (driver_imgs_df.classname).value_counts()
PlotClassFrequency(class_counts)
DescribeImageData(class_counts)

subject_counts = (driver_imgs_df.subject).value_counts()
plt.figure(figsize=(15,4))
plt.bar(subject_counts.index,subject_counts)
plt.xlabel('subject')
plt.ylabel('count')
plt.title('Number of Images per Subject')
plt.show()
DescribeImageData(subject_counts)

pd.isnull(driver_imgs_df).sum()

np.random.seed(0)
myarray = np.random.permutation(driver_imgs_df)
driver_imgs_df = pd.DataFrame(data = myarray, columns=['subject', 'classname', 'img'])

d = {'c0': 0, 'c1': 1, 'c2': 2, 'c3': 3, 'c4': 4, 'c5': 5, 'c6': 6, 'c7': 7, 'c8': 8, 'c9': 9}
driver_imgs_df.classname = driver_imgs_df.classname.map(d)

CreateImgArray(64, 64, 3, driver_imgs_df, 'train')

X = np.load('X_train_64_64.npy')
X.shape

Y = np.load('Y_train_64_64.npy')
Y.shape

(X == 0).sum()

PlotClassFrequency(pd.DataFrame(Y)[0].value_counts())

X_scaled = Rescale(X)

PrintImage(X_scaled, 2, Y = Y.reshape(-1,1))

def identity_block(X, f, filters, stage, block, init):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = init)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = init)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = init)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X

def convolutional_block(X, f, filters, stage, block, init, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = init)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = init)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c', kernel_initializer = init)(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1', kernel_initializer = init)(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X

def ResNet50(input_shape = (64, 64, 3), classes = 10, init = glorot_uniform(seed=0)):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = init)(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1, init = init)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b', init = init)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c', init = init)

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128,128,512], stage = 3, block='a', s = 2, init = init)
    X = identity_block(X, 3, [128,128,512], stage=3, block='b', init = init)
    X = identity_block(X, 3, [128,128,512], stage=3, block='c', init = init)
    X = identity_block(X, 3, [128,128,512], stage=3, block='d', init = init)

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2, init = init)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b', init = init)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c', init = init)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d', init = init)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e', init = init)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f', init = init)

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2, init = init)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b', init = init)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c', init = init)

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), name = 'avg_pool')(X)
    
    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = init)(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
    
    return model

# Normalize image vectors
X_train = X/255

# Convert training and test labels to one hot matrices
#Y = convert_to_one_hot(Y.astype(int), 10).T
Y_train = np.expand_dims(Y.astype(int), -1)

print ("number of training examples = " + str(X_train.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))

scores = LOGO(X_train, Y_train, group = driver_imgs_df['subject'],
              model_name = ResNet50, input_shape = (64, 64, 3), classes = 10, 
              init = glorot_uniform(seed=0), optimizer = 'adam', metrics = 'accuracy',
              epochs = 2, batch_size = 32)

plt.figure(figsize=(15,4))
plt.bar(scores.index, scores.loc[:,'Test_acc'].sort_values(ascending=False))
plt.yticks(np.arange(0, 110, 10.0))
plt.show()

scores.describe()

print("Train acc: {:.2f}. Dev. acc: {:.2f}".format(scores['Train_acc'].mean(), scores['Test_acc'].mean()))
print("Train loss: {:.2f}. Dev. loss: {:.2f}".format(scores['Train_loss'].mean(), scores['Test_loss'].mean()))

scores = LOGO(X_train, Y_train, group = driver_imgs_df['subject'],
              model_name = ResNet50, input_shape = (64, 64, 3), classes = 10, 
              init = glorot_uniform(seed=0), optimizer = 'adam', metrics = 'accuracy',
              epochs = 5, batch_size = 32)

print("Train acc: {:.2f}. Dev. acc: {:.2f}".format(scores['Train_acc'].mean(), scores['Test_acc'].mean()))
print("Train loss: {:.2f}. Dev. loss: {:.2f}".format(scores['Train_loss'].mean(), scores['Test_loss'].mean()))

scores = LOGO(X_train, Y_train, group = driver_imgs_df['subject'],
              model_name = ResNet50, input_shape = (64, 64, 3), classes = 10, 
              init = glorot_uniform(seed=0), optimizer = 'adam', metrics = 'accuracy',
              epochs = 10, batch_size = 32)

print("Train acc: {:.2f}. Dev. acc: {:.2f}".format(scores['Train_acc'].mean(), scores['Test_acc'].mean()))
print("Train loss: {:.2f}. Dev. loss: {:.2f}".format(scores['Train_loss'].mean(), scores['Test_loss'].mean()))

model = ResNet50(input_shape = (64, 64, 3), classes = 10)
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs = 10, batch_size = 32)

save_model(model, 'e10.h5');

model = load_model('e10.h5')

holdout_imgs_df = pd.read_csv('test_file_names.csv')
holdout_imgs_df.rename(columns={"imagename": "img"}, inplace = True)

CreateImgArray(64, 64, 3, holdout_imgs_df, 'test')

X_holdout = np.load('X_test_64_64.npy')
X_holdout.shape

probabilities = model.predict(X_holdout, batch_size = 32)

np.savetxt("test_results.csv", probabilities, delimiter=",")

X_holdout_scaled = Rescale(X_holdout)

index = 50000
PrintImage(X_holdout_scaled, index = index, Y = probabilities)
print('y_pred = ' + str(probabilities[index].argmax()))



