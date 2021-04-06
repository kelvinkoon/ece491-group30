import numpy as np
import cv2
import sys
import datetime
sys.path.append('../')
from acl_model import Model
from acl_resource import AclResource

## VINCENT IMPORTS
import os
from atlas_utils.camera import Camera
from atlas_utils import presenteragent
from atlas_utils.acl_image import AclImage
import acl

PRESENTER_CONF="presenterserver.conf"
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720

def execute():
    model_name_head_pose = 'head_pose_estimation'
    model_name_face_det = 'face_detection'

    # initialize acl resource
    acl_resource = AclResource()
    acl_resource.init()

    # load offline model for face detection
    MODEL_PATH = model_name_face_det + ".om"
    model_face = Model(acl_resource, MODEL_PATH)

    #load offline model for head pose estimation
    MODEL_PATH = model_name_head_pose + ".om"
    model_head_pose = Model(acl_resource, MODEL_PATH)

    ## Get Input ##
    # Initialize Camera
    cap = Camera(id = 0, fps = 10)

    ## Set Output ##
    # open the presenter channel
    chan = presenteragent.presenter_channel.open_channel(PRESENTER_CONF)
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
        img_original = cv2.flip(img_original,-1)
        input_image = PreProcessing_face(img_original)

        face_flag = False
        #face model inference and post processing
        try:
            resultList_face  = model_face.execute([input_image]).copy()
            xmin, ymin, xmax, ymax = PostProcessing_face(img_original, resultList_face)
            bbox_list = [xmin, ymin, xmax, ymax]
            face_flag = True
        except:
            print('No face detected')
            
        
        if face_flag is True:
            # # Preprocessing head pose estimation
            input_image = PreProcessing_head(img_original, bbox_list)
            # head pose estimation model inference
            try: 
                resultList_head = model_head_pose.execute([input_image]).copy()	
            except Exception as e:
                print('No head pose estimation output')

            #post processing to obtain coordinates for lines drawing
            facepointList, head_status_string, canvas = PostProcessing_head(resultList_head, bbox_list, img_original)
            # print('Head angles:', resultList_head[2])
            print('Pose:', head_status_string)
        else:
            canvas = img_original
        
        
        # ## Present Result ##
        # # convert to jpeg image for presenter server display
        _,jpeg_image = cv2.imencode('.jpg', canvas)
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

    execute()
