from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils.image_utils import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame,faceNet,maskNet):
    (h,w) = frame.shape[:2]
    
    
    blob = cv2.dnn.blobFromImage(frame,1.0,(224,224),(104.0,177.0,123.0))
    #Passing blob thorugh the network and obtaining the face detection
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)
    
    faces=[]
    locs=[]
    preds=[]
    
    #loop over the detections
    for i in range(0,detections.shape[2]):
        
        #extract confidence i.e probability associated with detection
        confidence = detections[0,0,i,2]
        
        #Filtering out weak detections by ensuring the confidence is greater than minimum confidence
        if confidence>0.5:
            
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY) = box.astype("int")
            
            (startX,startY) = (max(0,startX),max(0,startY))
            (endX,endY) = (min(w-1,endX),min(h-1,endY))
            
            face = frame[startY:endY,startX:endX]
            face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face = cv2.resize(face,(224,224))
            face = img_to_array(face)
            face = preprocess_input(face)
        
            faces.append(face)
            locs.append((startX,startY,endX,endY))
    
    if len(faces) >0:
        faces = np.array(faces,dtype="float32")
        preds = maskNet.predict(faces,batch_size = 32)
    
    #Reutrns face location and prediciton of wearing the mask
    return(locs,preds)
     
#Loading serialized face detector model from disk
prototxtPath = r"D:\Study\Projects\Mask Detection\face_detector\deploy.prototxt"
weightsPath = r"D:\Study\Projects\Mask Detection\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath,weightsPath)  # dnn: Deep Neural Network

#Lodaing Mask Detection Model from disk
maskNet = load_model("Mask_Detection.model")

#initialize video stream
print("Starting the camera...")
# vs = VideoStream(src="http://192.168.1.6:4747/video").start() For Mobile Camera
vs = VideoStream(src=0).start() #For default laptop camera


#loop over each frame captured using camera
while True:
    frame = vs.read()
    frame = imutils.resize(frame,width=1000,height=1000)
    (locs,preds) = detect_and_predict_mask(frame,faceNet,maskNet)
    
    for (box,pred) in zip(locs,preds):
        (startX,startY,endX,endY) = box
        (mask,withoutmask) = pred
        
        label = "Mask" if mask>withoutmask else "No Mask"
        color = (0,255,0) if label == "Mask" else (0,0,255)
        
        label = "{}:{:.2f}%".format(label,max(mask,withoutmask)*100)
        
        cv2.putText(frame,label,(startX,startY-10),cv2.FONT_HERSHEY_PLAIN,1,color,2)
        cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
        
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
    