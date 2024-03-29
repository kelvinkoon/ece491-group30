# Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
STATUS_DISCONNECT = 0
STATUS_CONNECTED = 1
STATUS_OPEN_CH_REQUEST = 2
STATUS_OPENED = 3
STATUS_EXITING = 4
STATUS_EXITTED = 5

CONTENT_TYPE_IMAGE = 0
CONTENT_TYPE_VIDEO = 1

STATUS_OK = 0
STATUS_ERROR = 1


class Point():
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class Box():
    def __init__(self, lt, rb):
        self.lt = Point(lt)
        self.rb = Point(rb)

    def box_valid(self):
        return ((self.lt.x >= 0)
                and (self.lt.y >= 0)
                and (self.rb.x >= self.lt.x)
                and (self.rb.y >= self.lt.y))

class ObjectDetectionResult():
    def __init__(self, ltx=0, lty=0, rbx=0, rby=0, text=None):
        self.object_class = 0
        self.confidence = 0
        self.box = Box((ltx, lty), (rbx, rby))
        self.result_text = text

