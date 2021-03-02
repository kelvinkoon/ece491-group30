import random
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

MODEL_PATH = "../model/body_pose.om"
BODYPOSE_CONF="../body_pose.conf"
MODEL_PATH_HEAD_POSE = 'head_pose_estimation.om'
MODEL_PATH_FACE = 'face_detection.om'
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
DATA_PATH = '../test_video/turn_right.mp4'
GO1_PATH = '../reference_pose/go1.jpg'
GO2_PATH = '../reference_pose/go2.jpg'
GO3_PATH = '../reference_pose/go3.jpg'
GO4_PATH = '../reference_pose/go4.jpg'
LEFT1_PATH = '../reference_pose/left1.jpg'
LEFT2_PATH = '../reference_pose/left2.jpg'
RIGHT1_PATH = '../reference_pose/right1.jpg'
RIGHT2_PATH = '../reference_pose/right2.jpg'
STOP_PATH = '../reference_pose/stop1.jpg'


class StateMachine:
    def __init__(self): 
        self.state = "idle"    # 初始状态
        self.temp=5000

    def idle(self, result, headpose_result):
        print(headpose_result)
        if headpose_result == "Head turn left slightly":
            self.state="stop"
            self.temp=5000
        else:
            self.state="idle"
    
    def stop(self, result, headpose_result):
        if(result=="stop"):
            print("the video is stop")
        elif(result=="left1"):
            self.state="left1"
            self.temp=5000
        elif(result=="right1"):
            self.temp=5000
            self.state="right1"
        elif(result=="go1"):
            self.temp=5000
            self.state="go1"

    def left1(self, result, headpose_result):
        if(result=="left2"):
            self.state="left2"
        elif(self.temp==0):
            self.state="stop"
        else:
            self.temp=self.temp-1
    
    def left2(self, result, headpose_result):
            self.state="terminate"
            print("the video is going left")

        
    def right1(self, result, headpose_result):
        if(result=="right2"):
            self.state="right2"
            print("the video is going right")
        elif(self.temp==0):
            self.state="stop"
        else:
            self.temp=self.temp-1
    
    def right2(self, result, headpose_result):
        self.state="terminate"
        print("the video is going right")
            
    def go1(self, result, headpose_result): 
        if(result=="go2"):
            self.state="go2"
            self.temp=5000
        elif(self.temp==0):
            self.state="stop"
        else:
            self.temp=self.temp-1
    
    def go2(self, result, headpose_result): 
        if(result=="go3"):
            self.state="go3"
            self.temp=5000
        elif(self.temp==0):
            self.state="stop"
        else:
            self.temp=self.temp-1
    
    def go3(self, result, headpose_result): 
        if(result=="go4"):
            self.state="terminate"
            print("the video is go")
        elif(self.temp==0):
            self.state="stop"
        else:
            self.temp=self.temp-1

    def terminate(self, result, headpose_result):
        exit(1)
        pass

    statelist = {
    "idle":idle,
    "stop":stop,
    "left1":left1,
    "left2":left2,
    "right1":right1,
    "right2":right2,
    "go1":go1,
    "go2":go2,
    "go3":go3,
    "terminate":terminate
    }       
    def staterunner(self, result, headpose_result):
        privousState=self.state
        self.statelist.get(self.state)(self, result, headpose_result)
        # self.statelist.get(self.state)
        print("the current state is",privousState,"  the result is ",result, "  the next state is",self.state)

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

def execute(model_path, frames_input_src, output_dir, is_presenter_server):

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
    resultList=[]
    ## Get Input ##
    # Read the video input using OpenCV
    cap = cv2.VideoCapture(frames_input_src)


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
    # Initialize count
    countgo=0
    countleft=0
    countright=0
    countstop=0


    ## Set Output ##
    if is_presenter_server:
        # if using presenter server, then open the presenter channel
        chan = presenteragent.presenter_channel.open_channel(BODYPOSE_CONF)
        if chan == None:
            print("Open presenter channel failed")
            return
    else:
        # if saving result as video file (mp4), then set the output video writer using opencv
        video_output_path = '{}/demo-{}-{}.mp4'.format(output_dir, os.path.basename(frames_input_src), str(random.randint(1, 100001)))
        video_writer = cv2.VideoWriter(video_output_path, 0x7634706d, 25,
                                                (1280, 720))
        if video_writer == None:
            print('Error: cannot get video writer from openCV')

    predict=StateMachine()

    while(cap.isOpened()):
        ## Read one frame of the input video ## 
        ret, img_original = cap.read()

        if not ret:
            print('Cannot read more, Reach the end of video')
            break

        ## HEAD POSE BEGIN
        # Camera Input (YUV) to RGB Image
        # image_byte = img_original.tobytes()
        # image_array = np.frombuffer(image_byte, dtype=np.uint8)
        # img_original_face = YUVtoRGB(image_array)
        # img_original_face = cv2.flip(img_original_face,1)

        img_original_face = img_original
        input_image = PreProcessing_face(img_original_face)

        face_flag = False
        #face model inference and post processing
        try:
            resultList_face  = model_face.execute([input_image]).copy()
            xmin, ymin, xmax, ymax = PostProcessing_face(img_original_face, resultList_face)
            bbox_list = [xmin, ymin, xmax, ymax]
            face_flag = True
        except:
            print('No face detected')
            
        head_status_string = "No output"
        if face_flag is True:
            # # Preprocessing head pose estimation
            input_image = PreProcessing_head(img_original_face, bbox_list)
            # head pose estimation model inference
            try: 
                resultList_head = model_head_pose.execute([input_image]).copy()	
            except Exception as e:
                print('No head pose estimation output')

            #post processing to obtain coordinates for lines drawing
            facepointList, head_status_string, canvas = PostProcessing_head(resultList_head, bbox_list, img_original_face)
            # print('Head angles:', resultList_head[2])
            print('Pose:', head_status_string)
        # else:
            # canvas = img_original_face

        headpose_result = head_status_string

        ## HEAD POSE END

        ## Model Prediction ##
        # model_processor.predict: processing + model inference + postprocessing
        # canvas: the picture overlayed with human body joints and limbs
        canvas_input, joint_list_input = model_processor.predict(img_original)
        
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

        
        #list_result=min(dif1,dif2,dif3,dif4,dif5,dif6,dif7,dif8,dif9)
        list_result=[dif1,dif2,dif3,dif4,dif5,dif6,dif7,dif8,dif9]
        decode_list={
            0:"go1",
            1:"go2",
            2:"go3",
            3:"go4",
            4:"left1",
            5:"left2",
            6:"right1",
            7:"right2",
            8:"stop",
        }

        result=decode_list[list_result.index(min(list_result))]
        
        resultList.append(result)
        predict.staterunner(result,headpose_result)
        ## Present Result ##
        if is_presenter_server:
            # convert to jpeg image for presenter server display
            _,jpeg_image = cv2.imencode('.jpg',canvas_input)
            # construct AclImage object for presenter server
            jpeg_image = AclImage(jpeg_image, img_original.shape[0], img_original.shape[1], jpeg_image.size)
            # send to presenter server
            chan.send_detection_data(img_original.shape[0], img_original.shape[1], jpeg_image, [])

        else:
            # save to video
            video_writer.write(canvas_input)
    print(resultList)
    # release the resources
    cap.release()
    if not is_presenter_server:
        video_writer.release()






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
            print('BBOX:', xmin, ymin, xmax, ymax)
            cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),1)
        else:
            continue
    print("detected bbox num:", bbox_num)
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
        head_status_string = head_status_string + " Good posture"
    return head_status_string

def PostProcessing_head(resultList, boxList, image):
    resultList.append([resultList[1][0][0] * 50, resultList[1][0][1] * 50, resultList[1][0][2] * 50])
    HeadPosePoint = []
    facepointList = []
    box_width = boxList[2] - boxList[0] 
    box_height = boxList[3] - boxList[1]
    box_width = box_width
    box_height = box_height
    print('box width:', box_width)
    print('box height:', box_height)
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

    cv2.imwrite('out/result_out.jpg', canvas)
    return facepointList, head_status_string, canvas
    

if __name__ == '__main__':   

    description = 'Load a model for human pose estimation'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', type=str, default=MODEL_PATH)
    parser.add_argument('--frames_input_src', type=str,default=DATA_PATH, help="Directory path for video.")
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Output Path")
    parser.add_argument('--is_presenter_server', type=bool, default=False, help="Display on presenter server or save to a video mp4 file (T/F)")
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    execute(args.model, args.frames_input_src, args.output_dir, args.is_presenter_server)
