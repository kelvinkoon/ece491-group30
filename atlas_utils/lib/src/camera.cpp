 /**
 * Copyright 2021 Huawei Technologies Co., Ltd.
 * Copyright 2021 HiSilicon Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#include <memory>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include "atlas_utils_common.h"
#include "camera.h"

using namespace std;

extern "C" {
#include "peripheral_api.h"
#include "camera.h"

CameraManager g_CameraMgr;

int CameraInit(int id, int fps, int width, int height) {   
	if (!g_CameraMgr.hwInited) {
		MediaLibInit();
		g_CameraMgr.hwInited = 1;
	}	

    Camera& cap = CAMERA(id);
	cap.frameSize = YUV420SP_SIZE(width, height);
    cap.id = id;
	cap.fps = fps;
	cap.width = width;
	cap.height = height;
	cap.inited = true;

	return STATUS_OK;
}

int ConfigCamera(int id, int fps, int width, int height) {
	int ret = SetCameraProperty(id, CAMERA_PROP_FPS, &fps);
	if (ret == LIBMEDIA_STATUS_FAILED) {
		ASC_LOG_ERROR("Set camera fps failed");
		return STATUS_ERROR;
	}

	CameraResolution resolution;
	resolution.width = width;
	resolution.height = height;
	ret = SetCameraProperty(id, CAMERA_PROP_RESOLUTION,	&resolution);
	if (ret == LIBMEDIA_STATUS_FAILED) {
		ASC_LOG_ERROR("Set camera resolution failed");
		return STATUS_ERROR;
	}

	CameraCapMode mode = CAMERA_CAP_ACTIVE;
	ret = SetCameraProperty(id, CAMERA_PROP_CAP_MODE, &mode);
	if (ret == LIBMEDIA_STATUS_FAILED) {
		ASC_LOG_ERROR("Set camera mode:%d failed", mode);
		return STATUS_ERROR;
	}

	return STATUS_OK;
}

int OpenCameraEx(int id, int fps, int width, int height) {
    if ((id < 0) || (id >= CAMERA_NUM)) {
		ASC_LOG_ERROR("Open camera failed for invalid id %d", id);
		return STATUS_ERROR;
	}

	if (!CAMERA(id).inited) {
		CameraInit(id, fps, width, height);
	}

	CameraStatus status = QueryCameraStatus(id);
	if (status == CAMERA_STATUS_CLOSED){
		// Open Camera
		if (LIBMEDIA_STATUS_FAILED == OpenCamera(id)) {
			ASC_LOG_ERROR("Camera%d closed, and open failed.", id);
			return STATUS_ERROR;
		}
	} else if (status != CAMERA_STATUS_OPEN) {
		ASC_LOG_ERROR("Invalid camera%d status %d", id, status);
		return STATUS_ERROR;
	}

	//Set camera property
	if (STATUS_OK != ConfigCamera(id, fps, width, height)) {
		CloseCamera(id);
		ASC_LOG_ERROR("Set camera%d property failed", id);
		return STATUS_ERROR;
	}

    ASC_LOG_INFO("Open camera %d success", id);

	return STATUS_OK;
}

int ReadCameraFrame(int id, CameraOutput& frame) {
	int size = CAMERA(id).frameSize;
	void* data = nullptr;
	auto aclRet = acldvppMalloc(&data, size);
    if (aclRet != ACL_ERROR_NONE) {
        ASC_LOG_ERROR("acl malloc dvpp data failed, dataSize=%u, ret=%d", 
                      size, aclRet);
        return STATUS_ERROR;
    }

	int ret = ReadFrameFromCamera(id, (void*)data, (int *)&size);
	if ((ret == LIBMEDIA_STATUS_FAILED) || 
	    (size != CAMERA(id).frameSize)) {
		acldvppFree(data);
		ASC_LOG_ERROR("Get image from camera %d failed, size %d", id, size);
		return STATUS_ERROR;
	}
	frame.size = size;
	frame.data = (uint8_t*)data;
    ASC_LOG_INFO("cpp image ptr 0x%x", data);
	return STATUS_OK;	
}

int CloseCameraEx(int cameraId) {
	if (LIBMEDIA_STATUS_FAILED == CloseCamera(cameraId)) {
		ASC_LOG_ERROR("Close camera %d failed", cameraId);
		return STATUS_ERROR;
	}

	return STATUS_OK;
}


}
