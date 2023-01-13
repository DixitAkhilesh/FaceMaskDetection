from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import AveragePooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils.image_utils import img_to_array
from keras.utils.image_utils import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import os

#initial learning rate, lower the rate nicer the accuracy
INIT_LR = 1e-5
Epochs = 30
BS = 64 #Batch size (Number of images to be considered in one batch)

DIRECTORY = r'D:\Study\Projects\Mask Detection\Dataset'
CATEGORIES = ['with_mask','without_mask']

print("Loading images...")

#Storing Images in the form of list
data = []

#Storing labels of images i.e with_mask or without_mask in list
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path): #All images are listed
        imgage_path = os.path.join(path,img)
        image = load_img(imgage_path,target_size=(224,224)) #(224,224) is height and width of image
        image = img_to_array(image)
        image = preprocess_input(image) #Used for mobilenet
        
        data.append(image)
        labels.append(category)

#Here our iamge is converted to int but labels are still strings with with_mask or without_mask values
#So we convert them into Binary Values        
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#Now we convert data and labels lists into arrays
data = np.array(data,dtype="float32")
labels = np.array(labels)

#Split the data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data,labels,train_size = 0.7,stratify=labels)

#Constructing the training image generator fro data augmentation
aug = ImageDataGenerator(
            rotation_range = 20,  
            zoom_range=0.15,
            height_shift_range=0.15,
            width_shift_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest")
#ImageDataGenerator is used to generate many images from one single images by slightly changing the original image

baseModel = MobileNetV2(weights= 'imagenet',include_top=False, input_tensor= Input(shape=(224,224,3))) 
#imagenet contains predefined functions for image processing and hence weight is set to imagenet
# #Here 3 denotes 3 channels i,e r,g,b i.e all colors are used as the images are colored images

#Constructing headModel object and passing baseModel output
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128,activation="relu")(headModel) #relu - rectified linear activation function
#relu is used for non linear usecases and is mainly used while image preprocessing  
headModel = Dropout(0.5)(headModel)
headModel = Dense(2,activation="softmax")(headModel) #we have 2 layered output with_mask and without_mask, softmax is good for visualization as it uses binary values 

#Calling the model function
model = Model(inputs= baseModel.input,outputs=headModel)

#loop over all layers in the baseModel and freeze them so
#they will not be updated during first training process

for layer in baseModel.layers:
    layer.trainable = False
    
#Compiling the Model
print("Compiling Model...")
opt = Adam(lr= INIT_LR,decay=INIT_LR/Epochs) #Adam optimizer is also a method which is widely used for image processing
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])#Tracking Accuracy matrix

print("Training head...")
H = model.fit(
        aug.flow(trainX,trainY,batch_size=BS),
        steps_per_epoch=len(trainX)//BS,
        validation_data = (testX,testY),
        validation_steps = len(testX)//BS,
        epochs = Epochs
    )
#Increasing images by adding images from aug to have more images for greater accuracy


print("Evaluating Network...")
preIdxs = model.predict(testX,batch_size=BS)#Evaluating the model

#For each image in testing set, we need to find the index of label with corresponding largest predicted probability
preIdxs = np.argmax(preIdxs,axis = 1)

print(classification_report(testY.argmax(axis=1),preIdxs,target_names=lb.classes_))

print("Saving Mask Detection Model...")
model.save("Mask_Detection.model",save_format = "h5")