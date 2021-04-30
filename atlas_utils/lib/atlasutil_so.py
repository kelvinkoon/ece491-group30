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
import threading
import ctypes
import os


class _AtlasutilLib(object):
    _instance_lock = threading.Lock()
    lib = ctypes.CDLL(os.path.dirname(os.path.abspath(__file__)) + '/libatlasutil.so')

    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        if not hasattr(_AtlasutilLib, "_instance"):
            with _AtlasutilLib._instance_lock:
                if not hasattr(_AtlasutilLib, "_instance"):
                    _AtlasutilLib._instance = object.__new__(cls)
        return _AtlasutilLib._instance

libatlas = _AtlasutilLib.lib
