#drydenwiebe

#import packages
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import time
import cv2
import numpy
from numpy import ndarray
import pybase64
import base64
from PIL import Image, ImageFile
import requests

#initialize the minimum area for the room to be considered "Occupied"
global minArea
minArea = 1000

#initialize the text as unoccupied
#global text
#text = "Unoccupied"

faceText = "No Face Detected"

# api-endpoint
global API_ENDPOINT
API_ENDPOINT = ""

#initialize the first frame as None
global firstFrame
firstFrame = None

#intialize the background frame sampling rate
global firstFrameRefresh
firstFrameRefresh = 100

#keep track of the total number of saves to the disk
global totalSaves

global cycleCount
cycleCount = 0

#initialie the framerate
global frameRate
frameRate = 1000

#load the training date into the CascadeClassifier
try:
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except:
    print("Error setting up classifier")
    raise

print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

#read the data from the picamera and then send it to the server
def read_and_send():
    while True:
        frame = vs.read()
        #save the resizing for a later decision
        #frame = imutils.resize(frame, height= 500, width=500)

        #break if the video stream has issues reading
        if frame is None:
            break

        #Make a gray and blurred verstion of the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayBlur = cv2.GaussianBlur(gray, (21, 21), 0)

        global firstFrame
        if firstFrame is None:
            firstFrame = grayBlur
            continue

        global cycleCount
        cycleCount = cycleCount + 1
        global firstFrameRefresh
        if cycleCount % firstFrameRefresh == 0:
            cycleCount = 0
            firstFrame = grayBlur
    
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 10)
            faceText = "Face Detected"

        if is_empty(faces):
            faceText = "No Face Detected"
            
        cv2.imshow("Face Detection", frame)

        # compute the absolute difference between the current frame and first frame
        frameDelta = cv2.absdiff(firstFrame, grayBlur)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
        # dilate the thresholded image to fill in holes, then find contours on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
 
        for c in contours:
            global minArea
            if cv2.contourArea(c) < minArea:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #global text
            #text = "Occupied"

        #if is_empty(contours):
            #global text
            #text = "Unoccupied"
            
        # draw the text and timestamp on the frame
        #global text
        #cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #cv2.putText(frame, "Face Detection: {}".format(faceText), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # show the frame and record if the user presses a key
        #cv2.imshow("Baby Monitor", frame)
        #cv2.imshow("Thresh", thresh)
        #cv2.imshow("Frame Delta", frameDelta)
        #key = cv2.waitKey(1) & 0xFF

        #send the current image to the database via POST requests
        #this is defined below
        send_image(frame, faces, contours)
        
        # if the `q` key is pressed, break from the loop
        #if key == ord("q"):
                #break

        #assert frameRate > 0
        #time.sleep(1/frameRate)

#compress and encode image funciton
def encode_image(image):
    success, encoded_image = cv2.imencode('.jpeg', image)
    if success:
        return pybase64.b64encode(encoded_image).decode()
    else:
        return ''

#sends an image via a POST request
def send_image(image, faces, contours):
    if is_empty(faces):
        facesIsEmpty = 0
    else:
        facesIsEmpty = 1

    if is_empty(contours):
        contoursIsEmpty = 0
    else:
        contoursIsEmpty = 1

    #image = imutils.resize(image, width=200, height=200)

    encoded_image = encode_image(image)

    b64_src = 'data:image/png;base64,'

    encoded_image = b64_src + encoded_image
    
    data = {"image":encoded_image,"faceDetected":facesIsEmpty,"motionDetected":contoursIsEmpty}

    response = requests.post(API_ENDPOINT, data)

    if(response.status_code == 200):
        print("Success!")
    else:
        print("Failure!")

#is_empty wrapper function
def is_empty(anyStructure):
    if len(anyStructure) == 0:
        return True
    else:
        return False

#encoding function
#returns: a string of the encoded image
def encode(image):
    #we first convert the image to jpeg and compress it 
    #encoding paramaters
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    new_jpg = cv2.imdecode(encimg, 1)
    encoded_image = encode_image(new_jpg)
    #use for HTML decoding
    b64_src = 'data:image/png;base64,'
    image_source = b64_src + encoded_image
    #print(image_source)
    return image_source

#main loop
def videoDetectionStream():
    while True:
        frame = vs.read()
        #save the resizing for a later decision
        #frame = imutils.resize(frame, height= 500, width=500)

        #break if the video stream has issues reading
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayBlur = cv2.GaussianBlur(gray, (21, 21), 0)

        global firstFrame
        if firstFrame is None:
            firstFrame = grayBlur
            continue

        global cycleCount
        cycleCount = cycleCount + 1
        global firstFrameRefresh
        if cycleCount % firstFrameRefresh == 0:
            cycleCount = 0
            firstFrame = grayBlur
    
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 10)
            #global faceText
            #faceText = "Face Detected"

        #if is_empty(faces):
            #global faceText
            #faceText = "No Face Detected"
            
        #cv2.imshow("Face Detection", frame)

        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, grayBlur)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
 
        for c in contours:
            global minArea
            if cv2.contourArea(c) < minArea:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #global text
           # text = "Occupied"

        #if is_empty(contours):
            #global text
            #text = "Unoccupied"
            

        #draw the text and timestamp on the frame
        #global text
        #cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #cv2.putText(frame, "Face Detection: {}".format(faceText), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        #show the frame and record if the user presses a key
        #cv2.imshow("Baby Monitor", frame)
        #cv2.imshow("Thresh", thresh)
        #cv2.imshow("Frame Delta", frameDelta)
        #key = cv2.waitKey(1) & 0xFF

        scale_percent = 100 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        image = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        #send the current image to the database via POST requests
        send_image(image, faces, contours)
        
        # if the `q` key is pressed, break from the loop
        #if key == ord("q"):
            #break

        #assert frameRate > 0
        #time.sleep(1/frameRate)

try:
    try:
        videoDetectionStream()
    except:
        print("Unable to start video stream")
        raise
finally:
    # cleanup the camera and close any open windows
    cv2.destroyAllWindows()
    vs.stop()
