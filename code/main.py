import os
import cv2
import numpy as np
import math
import argparse
import sys
sys.path.append('..')
from model_processor import ModelProcessor
from atlas_utils.camera import Camera
from atlas_utils import presenteragent
from atlas_utils.acl_image import AclImage
import acl
from acl_model import Model
from acl_resource import AclResource
import copy
import uart

MODEL_PATH = "../model/body_pose.om"
MODEL_PATH_HEAD_POSE = '../model/head_pose_estimation.om'
MODEL_PATH_FACE = '../model/face_detection.om'
BODYPOSE_CONF="../body_pose.conf"
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720

LEFT1_PATH = '../reference_pose/left1.jpg'
LEFT2_PATH = '../reference_pose/left2.jpg'
RIGHT1_PATH = '../reference_pose/old/right1.jpg'
RIGHT2_PATH = '../reference_pose/old/right2.jpg'
STOP_PATH = '../reference_pose/stop1.jpg'


class StateMachine:
    def __init__(self): 
        self.state = "idle"    # 初始状态
        self.temp=5000

        ## UART BEGIN
        self.conn = uart.UART()
        self.conn.open()

    def idle(self, result, headpose_result):
        print(headpose_result)
        # if True:
        if headpose_result == "Head facing camera":
            self.state="stop"
            self.temp=5000
        else:
            self.state="idle"
    
    def stop(self, result, headpose_result):
        if(result=="stop"):
            self.conn.send_stop()
            print("the video is stop")
        elif(result=="left1"):
            self.state="left1"
            self.temp=5000
        elif(result=="right1"):
            self.temp=5000
            self.state="right1"

    def left1(self, result, headpose_result):
        if(result=="left2"):
            self.state="left2"
        elif(self.temp==0):
            self.state="stop"
        else:
            self.temp=self.temp-1
    
    def left2(self, result, headpose_result):
            self.state="terminate"
            self.conn.send_left()
            print("the video is going left2")

    def right1(self, result, headpose_result):
        if(result=="right2"):
            self.state="right2"
        elif(self.temp==0):
            self.state="stop"
        else:
            self.temp=self.temp-1
    
    def right2(self, result, headpose_result):
        self.state="terminate"
        self.conn.send_right()
        print("the video is going right2")
            
    def terminate(self, result, headpose_result):
        print("terminate")
        pass

    statelist = {
        "idle":idle,
        "stop":stop,
        "left1":left1,
        "left2":left2,
        "right1":right1,
        "right2":right2,
        "terminate":terminate
    }

    def staterunner(self, result, headpose_result):
        prevState=self.state
        self.statelist.get(self.state)(self, result, headpose_result)
        print(f'Current state: {prevState}, Result: {result}, Next State: {self.state}')


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

    angle[0]=math.acos(ratio1)*180/(math.pi)
    angle[1]=math.acos(ratio2)*180/(math.pi)
    angle[2]=math.acos(ratio3)*180/(math.pi)
    angle[3]=math.acos(ratio4)*180/(math.pi)

    #adjust the range of angle from pi to 2pi
    if np.cross(vec1_2,vec1_0)<0:
        angle[0]=360-angle[0]
    if np.cross(vec0_1,vec0_13)<0:
        angle[1]=360-angle[1]
    if np.cross(vec4_3,vec4_5)<0:
        angle[2]=360-angle[2]
    if np.cross(vec3_13,vec3_4)<0:
        angle[3]=360-angle[3] 
    return angle

def execute(model_path):

    ## Initialization ##
    #initialize acl runtime 
    acl_resource = AclResource()
    acl_resource.init()

    # load offline model for face detection
    model_face = Model(acl_resource, MODEL_PATH_FACE)
    model_head_pose = Model(acl_resource, MODEL_PATH_HEAD_POSE)
    
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
    last_five_frame_result = [] 

    # Initialize Camera
    cap = Camera(id = 0, fps = 10)

    # Read reference images
    img_left1 = cv2.imread(LEFT1_PATH)
    img_left2 = cv2.imread(LEFT2_PATH)
    img_right1 = cv2.imread(RIGHT1_PATH)
    img_right2 = cv2.imread(RIGHT2_PATH)
    img_stop = cv2.imread(STOP_PATH)
    # Get reference output
    canvas_left1,joint_list_left1 = model_processor.predict(img_left1)
    canvas_left2,joint_list_left2 = model_processor.predict(img_left2)
    canvas_right1,joint_list_right1 = model_processor.predict(img_right1)
    canvas_right2,joint_list_right2 = model_processor.predict(img_right2)
    canvas_stop,joint_list_stop = model_processor.predict(img_stop)
    # Get angles from reference images
    angle_left1=getangle(joint_list_left1)
    angle_left2=getangle(joint_list_left2)
    angle_right1=getangle(joint_list_right1)
    angle_right2=getangle(joint_list_right2)
    angle_stop=getangle(joint_list_stop)
    # Initialize count
    countleft=0
    countright=0
    countstop=0

    ## Presenter Server Output ##
    chan = presenteragent.presenter_channel.open_channel(BODYPOSE_CONF)
    if chan == None:
        print("Open presenter channel failed")
        return

    predict = StateMachine()

    while True:
        ## Read one frame of the input video ## 
        img_original = cap.read()

        if not img_original:
            print('Error: Camera read failed')
            break

        ## HEAD POSE BEGIN ##
        # Camera Input (YUV) to RGB Image
        image_byte = img_original.tobytes()
        image_array = np.frombuffer(image_byte, dtype=np.uint8) 
        img_original = YUVtoRGB(image_array)
        img_original = cv2.flip(img_original, -1)
        
        # Make copy of image for head model processing and body model processing 
        img_bodypose = copy.deepcopy(img_original)
        img_headpose = copy.deepcopy(img_original)
        
        ## Model Prediction ##
        # model_processor.predict: processing + model inference + postprocessing
        # canvas: the picture overlayed with human body joints and limbs
        # img_bodypose is modified with skeleton
        canvas, joint_list_input = model_processor.predict(img_bodypose)
        
        angle_input=getangle(joint_list_input)


        dif5=abs(angle_input-angle_left1)
        dif6=abs(angle_input-angle_left2)
        dif7=abs(angle_input-angle_right1)
        dif8=abs(angle_input-angle_right2)
        dif9=abs(angle_input-angle_stop)
        
        result = "invalid"
        # last_five_result = "invalid"
        if all( i < 25 for i in dif5):
            result = "left1"
        elif all( i < 25 for i in dif6):
            result = "left2"
        elif all( i < 25 for i in dif7):
            result = "right1"
        elif all( i < 25 for i in dif8):
            result = "right2"
        elif all( i < 25 for i in dif9):
            result = "stop"            
        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 100)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2

        cv2.putText(img_bodypose, result, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
    

        ## FACE DETECTION MODEL BEGIN ##
        input_image = PreProcessing_face(img_headpose)

        face_flag = False
        try:
            resultList_face  = model_face.execute([input_image]).copy()
            # draw bounding box on img_bodypose
            xmin, ymin, xmax, ymax = PostProcessing_face(img_bodypose, resultList_face)
            bbox_list = [xmin, ymin, xmax, ymax]
            face_flag = True
        except:
            print('No face detected')
        # FACE DETECTION MODEL END ##

        ## HEADPOSE BEGIN ##
        head_status_string = "No output"
        if face_flag is True:
            input_image = PreProcessing_head(img_headpose, bbox_list)
            try: 
                resultList_head = model_head_pose.execute([input_image]).copy()	
            except Exception as e:
                print('No head pose estimation output')

            # draw headpose points on image
            facepointList, head_status_string, canvas = PostProcessing_head(resultList_head, bbox_list, img_bodypose)
            print('Headpose:', head_status_string)

        headpose_result = head_status_string
        ## HEADPOSE END ##

        predict.staterunner(result,headpose_result)
        ## Present Result ##
        # convert to jpeg image for presenter server display
        _,jpeg_image = cv2.imencode('.jpg', img_bodypose)
        # construct AclImage object for presenter server
        jpeg_image = AclImage(jpeg_image, img_original.shape[0], img_original.shape[1], jpeg_image.size)
        # send to presenter server
        chan.send_detection_data(img_original.shape[0], img_original.shape[1], jpeg_image, [])

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

# face detection model preprocessing
def PreProcessing_face(image):
    image = cv2.resize(image, (300,300))
    image = image.astype('float32')
    image = np.transpose(image, (2, 0, 1)).copy()
    return image

# face detection model post processing
def PostProcessing_face(image, resultList, threshold=0.9):
    detections = resultList[1]
    bbox_num = 0
    bbox_list = []
    for i in range(detections.shape[1]):
        det_conf = detections[0,i,2]
        det_xmin = detections[0,i,3]
        det_ymin = detections[0,i,4]
        det_xmax = detections[0,i,5]
        det_ymax = detections[0,i,6]
        bbox_width = det_xmax - det_xmin
        bbox_height = det_ymax - det_ymin
        if threshold <= det_conf and 1>=det_conf and bbox_width>0 and bbox_height > 0:
            bbox_num += 1
            xmin = int(round(det_xmin * image.shape[1]))
            ymin = int(round(det_ymin * image.shape[0]))
            xmax = int(round(det_xmax * image.shape[1]))
            ymax = int(round(det_ymax * image.shape[0]))
            # print('BBOX:', xmin, ymin, xmax, ymax)
            cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),1)
        else:
            continue
    #print("detected bbox num:", bbox_num)
    return [xmin, ymin, xmax, ymax]


# head pose estimation preprocessing
def PreProcessing_head(image, boxList):
    # convert to float type
    image = np.asarray(image, dtype=np.float32)
    # crop out detected face
    image = image[int(boxList[1]):int(boxList[3]),int(boxList[0]):int(boxList[2])]
    # resize to required input dimensions
    image = cv2.resize(image, (224, 224))
    # switch from NHWC to NCHW format
    image = np.transpose(image, (2, 0, 1)).copy()
    return image
    

# determine head pose from pitch, yaw, roll angle ranges
def head_status_get(resultList):
    # initialize
    fg_pitch = True
    fg_yaw = True
    fg_roll = True
    # find the viewing direction
    head_status_string = "Head"
    if resultList[2][0] < -20:
        head_status_string = head_status_string + " up"
    elif resultList[2][0] < -10:
        head_status_string = head_status_string + " up slightly"
    elif resultList[2][0] > 20:
        head_status_string = head_status_string + " down sharply"
    elif resultList[2][0] > 10:
        head_status_string = head_status_string + " down slightly"
    else:
        fg_pitch = False
    if resultList[2][1] < -23:
        head_status_string = head_status_string + " sharp left"
    elif resultList[2][1] < -10:
        head_status_string = head_status_string + " turn left slightly"
    elif resultList[2][1] > 23:
        head_status_string = head_status_string + " turn right"
    elif resultList[2][1] > 10:
        head_status_string = head_status_string + " turn right slightly"
    else:
        fg_yaw = False
    if resultList[2][2] < -20:
        head_status_string = head_status_string + " Swing right"
    elif resultList[2][2] < -10:
        head_status_string = head_status_string + " Swing right"
    elif resultList[2][2] > 20:
        head_status_string = head_status_string + " Swing left"
    elif resultList[2][2] > 10:
        head_status_string = head_status_string + " Swing left"
    else:
        fg_roll = False
    if fg_pitch is False and fg_yaw is False and fg_roll is False:
        head_status_string = head_status_string + " facing camera"
    return head_status_string

def PostProcessing_head(resultList, boxList, image):
    resultList.append([resultList[1][0][0] * 50, resultList[1][0][1] * 50, resultList[1][0][2] * 50])
    HeadPosePoint = []
    facepointList = []
    box_width = boxList[2] - boxList[0] 
    box_height = boxList[3] - boxList[1]
    box_width = box_width
    box_height = box_height
    # print('box width:', box_width)
    # print('box height:', box_height)
    for j in range(136):
        if j % 2 == 0:
            HeadPosePoint.append((1+resultList[0][0][j])/2  * box_width + boxList[0])
        else:
            HeadPosePoint.append((1+resultList[0][0][j])/2  * box_height + boxList[1])
        facepointList.append(HeadPosePoint)
    for j in range(136):
        if j % 2 == 0:
            canvas = cv2.circle(image, (int(facepointList[0][j]), int(facepointList[0][j+1])), 
                                radius=5, color=(255, 0, 0), thickness=2)
    head_status_string = head_status_get(resultList)

    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 50)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(canvas, head_status_string, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    # cv2.imwrite('out/result_out.jpg', canvas)
    return facepointList, head_status_string, canvas
    

if __name__ == '__main__':   

    description = 'Load a model for human pose estimation'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', type=str, default=MODEL_PATH)
    args = parser.parse_args()

    execute(args.model)
