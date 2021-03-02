import os
import cv2
import numpy as np
import math
import argparse
import sys
sys.path.append('..')
from model_processor import ModelProcessor
import acl
from acl_resource import AclResource

MODEL_PATH = "../model/body_pose.om"
DATA_PATH = '../reference_pose/left1.jpg'
GO1_PATH = '../reference_pose/go1.jpg'
GO2_PATH = '../reference_pose/go2.jpg'
GO3_PATH = '../reference_pose/go3.jpg'
GO4_PATH = '../reference_pose/go4.jpg'
LEFT1_PATH = '../reference_pose/left1.jpg'
LEFT2_PATH = '../reference_pose/left2.jpg'
RIGHT1_PATH = '../reference_pose/right1.jpg'
RIGHT2_PATH = '../reference_pose/right2.jpg'
STOP_PATH = '../reference_pose/stop.jpg'

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
    angle[0]=math.acos(np.dot(vec1_2,vec1_0)/math.sqrt(np.dot(vec1_2,vec1_2)*np.dot(vec1_0,vec1_0)))
    angle[1]=math.acos(np.dot(vec0_1,vec0_13)/math.sqrt(np.dot(vec0_1,vec0_1)*np.dot(vec0_13,vec0_13)))
    angle[2]=math.acos(np.dot(vec4_5,vec4_3)/math.sqrt(np.dot(vec4_5,vec4_5)*np.dot(vec4_3,vec4_3)))
    angle[3]=math.acos(np.dot(vec3_4,vec3_13)/math.sqrt(np.dot(vec3_4,vec3_4)*np.dot(vec3_13,vec3_13)))

    return angle


def execute(model_path, frames_input_src, output_dir):
    

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
    # Read the image input using OpenCV
    img_original = cv2.imread(args.frames_input_src)
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
    ## Model Prediction ##
    # model_processor.predict: processing + model inference + postprocessing
    # canvas: the picture overlayed with human body joints and limbs
    canvas_input,joint_list_input = model_processor.predict(img_original)
    canvas_go1,joint_list_go1 = model_processor.predict(img_go1)
    canvas_go2,joint_list_go2 = model_processor.predict(img_go2)
    canvas_go3,joint_list_go3 = model_processor.predict(img_go3)
    canvas_go4,joint_list_go4 = model_processor.predict(img_go4)
    canvas_left1,joint_list_left1 = model_processor.predict(img_left1)
    canvas_left2,joint_list_left2 = model_processor.predict(img_left2)
    canvas_right1,joint_list_right1 = model_processor.predict(img_right1)
    canvas_right2,joint_list_right2 = model_processor.predict(img_right2)
    canvas_stop,joint_list_stop = model_processor.predict(img_stop)

    angle_input=getangle(joint_list_input)
    angle_go1=getangle(joint_list_go1)
    angle_go2=getangle(joint_list_go2)
    angle_go3=getangle(joint_list_go3)
    angle_go4=getangle(joint_list_go4)
    angle_left1=getangle(joint_list_left1)
    angle_left2=getangle(joint_list_left2)
    angle_right1=getangle(joint_list_right1)
    angle_right2=getangle(joint_list_right2)
    angle_stop=getangle(joint_list_stop)
    print(angle_input)

    dif1=abs(np.sum(angle_input-angle_go1))
    dif2=abs(np.sum(angle_input-angle_go2))
    dif3=abs(np.sum(angle_input-angle_go3))
    dif4=abs(np.sum(angle_input-angle_go4))
    dif5=abs(np.sum(angle_input-angle_left1))
    dif6=abs(np.sum(angle_input-angle_left2))
    dif7=abs(np.sum(angle_input-angle_right1))
    dif8=abs(np.sum(angle_input-angle_right2))
    dif9=abs(np.sum(angle_input-angle_stop))
    
    pose=min(dif1,dif2,dif3,dif4,dif5,dif6,dif7,dif8,dif9)

    if pose==dif1 or pose==dif2 or pose==dif3 or pose==dif4:
        print('go straight')
    elif pose==dif5 or pose==dif6:
        print('turn left')
    elif pose==dif7 or pose==dif8:
        print('turn right')
    elif pose==dif9:
        print('stop')

    # Save the detected results
    cv2.imwrite(os.path.join(args.output_dir, 'Result_Pose.jpg'), canvas_input)
    

if __name__ == '__main__':   
    description = 'Load a model for human pose estimation'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', type=str, default=MODEL_PATH)
    parser.add_argument('--frames_input_src', type=str,default=DATA_PATH, help="Directory path for image")
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Output Path")

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    execute(args.model, args.frames_input_src, args.output_dir)





