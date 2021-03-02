import random
import os
import cv2
import numpy as np
import argparse
import sys
sys.path.append('..')
from model_processor import ModelProcessor
from atlas_utils.camera import Camera
from atlas_utils import presenteragent
from atlas_utils.acl_image import AclImage
import acl
from acl_resource import AclResource

MODEL_PATH = "../model/body_pose.om"
BODYPOSE_CONF="../body_pose.conf"
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
GO1_PATH = '../reference_pose/go1.jpg'
GO2_PATH = '../reference_pose/go2.jpg'
GO3_PATH = '../reference_pose/go3.jpg'
GO4_PATH = '../reference_pose/go4.jpg'
LEFT1_PATH = '../reference_pose/left1.jpg'
LEFT2_PATH = '../reference_pose/left2.jpg'
RIGHT1_PATH = '../reference_pose/right1.jpg'
RIGHT2_PATH = '../reference_pose/right2.jpg'
STOP_PATH = '../reference_pose/stop.jpg'

class StateMachine:
    def __init__(self): 
        self.state = "stop"    # 初始状态
        self.temp=5
    
    def stop(self, result):
        if(result=="stop")
            print("the video is stop")
        elif(reult=="left1")
            self.state="left1"
            self.temp=5
        elif(reult=="right1")
            self.temp=5
            self.state="right1"
        elif(reult=="go1")
            self.temp=5
            self.state="go1"

    def left1(self, result):
        if(result=="left2")
            self.state="stop"
            print("the video is going left")
        elif(self.temp==0)
            self.state="stop"
        else
            self.temp=self.temp-1
            
        
    def right1(self, result):
        if(result=="right2")
            self.state="stop"
            print("the video is going right")
        elif(self.temp==0)
            self.state="stop"
        else
            self.temp=self.temp-1
            
    def go1(self, result): 
        if(result=="go2")
            self.state="go2"
            self.temp=5
        elif(self.temp==0)
            self.state="stop"
        else
            self.temp=self.temp-1
    
    def go2(self, result): 
        if(result=="go3")
            self.state="go3"
            self.temp=5
        elif(self.temp==0)
            self.state="stop"
        else
            self.temp=self.temp-1
    
    def go3(self, result): 
        if(result=="go4")
            self.state="stop"
            print("the video is go")
        elif(self.temp==0)
            self.state="stop"
        else
            self.temp=self.temp-1
            
    def staterunner(self, result):
        statelist = {
        "stop":stop,
        "left1":left1,
        "right1":right1,
        "go1":go1,
        "go2":go2,
        "go3":go3,
        }
        statelist.get(self.state,"stop")(result) 

def getangle(point):
    #empty array for angles
    angle=np.zeros(4)

    #define vectors, e.g. vec1_2 is the vector from point 1 to point 2
    vec1_2=point[2]-point[1]
    vec1_0=point[0]-point[1]

    vec0_1=point[1]-point[0]
    vec0_13=point[13]-point[0]

    vec4_5=point[5]-point[4]
    vec4_3=point[3]-point[4]

    vec3_4=point[4]-point[3]
    vec3_13=point[13]-point[3]

    #calculate angles using dot product
    ratio1=round(np.dot(vec1_2,vec1_0)/math.sqrt(np.dot(vec1_2,vec1_2)*np.dot(vec1_0,vec1_0)), 10)
    ratio2=round(np.dot(vec0_1,vec0_13)/math.sqrt(np.dot(vec0_1,vec0_1)*np.dot(vec0_13,vec0_13)),10)
    ratio3=round(np.dot(vec4_5,vec4_3)/math.sqrt(np.dot(vec4_5,vec4_5)*np.dot(vec4_3,vec4_3)),10)
    ratio4=round(np.dot(vec3_4,vec3_13)/math.sqrt(np.dot(vec3_4,vec3_4)*np.dot(vec3_13,vec3_13)),10)

    angle[0]=math.acos(ratio1)
    angle[1]=math.acos(ratio2)
    angle[2]=math.acos(ratio3)
    angle[3]=math.acos(ratio4)

    return angle

def execute(model_path):

    ## Initialization ##
    #initialize acl runtime 
    acl_resource = AclResource()
    acl_resource.init()

    ## Prepare Model ##
    # parameters for model path and model inputs
    model_parameters = {
        'model_dir': model_path,
        'width': 368, # model input width      
        'height': 368, # model input height
    }
    # perpare model instance: init (loading model from file to memory)
    # model_processor: preprocessing + model inference + postprocessing
    model_processor = ModelProcessor(acl_resource, model_parameters)
    
    # Read reference images
    img_go1 = cv2.imread(GO1_PATH)
    img_go2 = cv2.imread(GO2_PATH)
    img_go3 = cv2.imread(GO3_PATH)
    img_go4 = cv2.imread(GO4_PATH)
    img_left1 = cv2.imread(LEFT1_PATH)
    img_left2 = cv2.imread(LEFT2_PATH)
    img_right1 = cv2.imread(RIGHT1_PATH)
    img_right2 = cv2.imread(RIGHT2_PATH)
    img_stop = cv2.imread(STOP_PATH)
    # Get reference output
    canvas_go1,joint_list_go1 = model_processor.predict(img_go1)
    canvas_go2,joint_list_go2 = model_processor.predict(img_go2)
    canvas_go3,joint_list_go3 = model_processor.predict(img_go3)
    canvas_go4,joint_list_go4 = model_processor.predict(img_go4)
    canvas_left1,joint_list_left1 = model_processor.predict(img_left1)
    canvas_left2,joint_list_left2 = model_processor.predict(img_left2)
    canvas_right1,joint_list_right1 = model_processor.predict(img_right1)
    canvas_right2,joint_list_right2 = model_processor.predict(img_right2)
    canvas_stop,joint_list_stop = model_processor.predict(img_stop)
    # Get angles from reference images
    angle_go1=getangle(joint_list_go1)
    angle_go2=getangle(joint_list_go2)
    angle_go3=getangle(joint_list_go3)
    angle_go4=getangle(joint_list_go4)
    angle_left1=getangle(joint_list_left1)
    angle_left2=getangle(joint_list_left2)
    angle_right1=getangle(joint_list_right1)
    angle_right2=getangle(joint_list_right2)
    angle_stop=getangle(joint_list_stop)

    ## Get Input ##
    # Initialize Camera
    cap = Camera(id = 0, fps = 10)

    ## Set Output ##
    # open the presenter channel
    chan = presenteragent.presenter_channel.open_channel(BODYPOSE_CONF)
    if chan == None:
        print("Open presenter channel failed")
        return



    while True:
        ## Read one frame from Camera ## 
        img_original = cap.read()
        if not img_original:
            print('Error: Camera read failed')
            break
        # Camera Input (YUV) to RGB Image
        image_byte = img_original.tobytes()
        image_array = np.frombuffer(image_byte, dtype=np.uint8)
        img_original = YUVtoRGB(image_array)
        img_original = cv2.flip(img_original,1)

        ## Model Prediction ##
        # model_processor.predict: processing + model inference + postprocessing
        # canvas: the picture overlayed with human body joints and limbs
        canvas_input,joint_list_input = model_processor.predict(img_original)
        angle_input=getangle(joint_list_input)

        dif1=abs(np.sum(angle_input-angle_go1))
        dif2=abs(np.sum(angle_input-angle_go2))
        dif3=abs(np.sum(angle_input-angle_go3))
        dif4=abs(np.sum(angle_input-angle_go4))
        dif5=abs(np.sum(angle_input-angle_left1))
        dif6=abs(np.sum(angle_input-angle_left2))
        dif7=abs(np.sum(angle_input-angle_right1))
        dif8=abs(np.sum(angle_input-angle_right2))
        dif9=abs(np.sum(angle_input-angle_stop))

        result=min(dif1,dif2,dif3,dif4,dif5,dif6,dif7,dif8,dif9)
        predict=StateMachine()
        predict.staterunner(result)

        ## Present Result ##
        # convert to jpeg image for presenter server display
        _,jpeg_image = cv2.imencode('.jpg',canvas)
        # construct AclImage object for presenter server
        jpeg_image = AclImage(jpeg_image, img_original.shape[0], img_original.shape[1], jpeg_image.size)
        # send to presenter server
        chan.send_detection_data(img_original.shape[0], img_original.shape[1], jpeg_image, [])

    # release the resources
    cap.release()

def YUVtoRGB(byteArray):
    e = 1280*720
    Y = byteArray[0:e]
    Y = np.reshape(Y, (720,1280))

    s = e
    V = byteArray[s::2]
    V = np.repeat(V, 2, 0)
    V = np.reshape(V, (360,1280))
    V = np.repeat(V, 2, 0)

    U = byteArray[s+1::2]
    U = np.repeat(U, 2, 0)
    U = np.reshape(U, (360,1280))
    U = np.repeat(U, 2, 0)

    RGBMatrix = (np.dstack([Y,U,V])).astype(np.uint8)
    RGBMatrix = cv2.cvtColor(RGBMatrix, cv2.COLOR_YUV2RGB, 3)
    return RGBMatrix
   

if __name__ == '__main__':   

    description = 'Load a model for human pose estimation'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', type=str, default=MODEL_PATH)
    args = parser.parse_args()

    execute(args.model)

