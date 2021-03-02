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
from acl_resource import AclResource

MODEL_PATH = "../model/body_pose.om"
BODYPOSE_CONF="../body_pose.conf"
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
DATA_PATH = '../test_video/turn_right.mp4'
GO1_PATH = '../reference_pose/go1.jpg'#
GO2_PATH = '../reference_pose/go2.jpg'#
GO3_PATH = '../reference_pose/go3.jpg'#
GO4_PATH = '../reference_pose/go4.jpg'#
LEFT1_PATH = '../reference_pose/left1.jpg'
LEFT2_PATH = '../reference_pose/left2.jpg'
RIGHT1_PATH = '../reference_pose/right1.jpg'
RIGHT2_PATH = '../reference_pose/right2.jpg'
STOP1_PATH = '../reference_pose/stop1.jpg'
#STOP2_PATH = '../reference_pose/stop2.jpg'

"""
Joints Explained
14 joints:
0-right shoulder, 1-right elbow, 2-right wrist, 3-left shoulder, 4-left elbow, 5-left wrist, 
6-right hip, 7-right knee, 8-right ankle, 9-left hip, 10-left knee, 11-left ankle, 
12-top of the head and 13-neck

                     12                     
                     |
                     |
               0-----13-----3
              /     / \      \
             1     /   \      4
            /     /     \      \
           2     6       9      5
                 |       |
                 7       10
                 |       |
                 8       11
"""
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
    img_stop1 = cv2.imread(STOP1_PATH)
    #img_stop2 = cv2.imread(STOP2_PATH)
    # Get reference output
    canvas_go1,joint_list_go1 = model_processor.predict(img_go1)
    canvas_go2,joint_list_go2 = model_processor.predict(img_go2)
    canvas_go3,joint_list_go3 = model_processor.predict(img_go3)
    canvas_go4,joint_list_go4 = model_processor.predict(img_go4)
    canvas_left1,joint_list_left1 = model_processor.predict(img_left1)
    canvas_left2,joint_list_left2 = model_processor.predict(img_left2)
    canvas_right1,joint_list_right1 = model_processor.predict(img_right1)
    canvas_right2,joint_list_right2 = model_processor.predict(img_right2)
    canvas_stop1,joint_list_stop1 = model_processor.predict(img_stop1)
    #canvas_stop2,joint_list_stop2 = model_processor.predict(img_stop2)
    # Get angles from reference images
    angle_go1=getangle(joint_list_go1)
    angle_go2=getangle(joint_list_go2)
    angle_go3=getangle(joint_list_go3)
    angle_go4=getangle(joint_list_go4)
    angle_left1=getangle(joint_list_left1)
    angle_left2=getangle(joint_list_left2)
    angle_right1=getangle(joint_list_right1)
    angle_right2=getangle(joint_list_right2)
    angle_stop1=getangle(joint_list_stop1)
    #angle_stop2=getangle(joint_list_stop2)
    # Initialize count
    countgo=0
    countleft=0
    countright=0
    countstop=0
    countinvalid=0
    countleft1=0
    countleft2=0
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


    while(cap.isOpened()):
        ## Read one frame of the input video ## 
        ret, img_original = cap.read()

        if not ret:
            print('Cannot read more, Reach the end of video')
            break

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
        dif9=abs(np.sum(angle_input-angle_stop1))
        #dif10=abs(np.sum(angle_input-angle_stop2))

        print(dif1,dif2,dif3,dif4,dif5,dif6,dif7,dif8,dif9)
        pose=min(dif1,dif2,dif3,dif4,dif5,dif6,dif7,dif8,dif9)

        if pose>0.5:
            countinvalid=countinvalid+1
        
        else:

            if pose==dif1 or pose==dif2 or pose==dif3 or pose==dif4:
                countgo=countgo+1
            elif pose==dif5 or pose==dif6:
                countleft=countleft+1
                if pose==dif5:
                    countleft1=countleft1+1
                else: 
                    countleft2=countleft2+1
            elif pose==dif7 or pose==dif8:
                countright=countright+1
            elif pose==dif9:
                countstop=countstop+1
            
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
    result=max(countgo,countleft,countright,countstop)
    print(countgo,countleft,countright,countstop,countinvalid,countleft1,countleft2)
    if result==countgo:
        print('go straight')
    elif result==countleft:
        print('turn left')
    elif result==countright:
        print('turn right')
    elif result==countstop:
        print('stop')

    # release the resources
    cap.release()
    if not is_presenter_server:
        video_writer.release()
   

    

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
