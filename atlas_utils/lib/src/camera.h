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
#ifndef _CAMERA_H
#define _CAMERA_H

#define CAMERA_NUM     (2)

#define CAMERA(i) (g_CameraMgr.cap[i])

struct CameraOutput {
	int size;
	uint8_t* data;
};

struct Camera {
	bool inited = false;
	int id = 255;
	int fps = 0;
	int width = 0;
	int height = 0;
	int frameSize = 0;

};

struct CameraManager {
	bool hwInited = 0;
	Camera cap[CAMERA_NUM];
};




#endif
