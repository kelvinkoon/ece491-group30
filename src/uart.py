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
# limitations under the License.# uart.py
import serial

class UART:

    # default port and default baudrate set
    def __init__(self, port="/dev/ttyAMA1", baudrate=115200):
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baudrate

    def open(self):
        self.ser.open()

    def close(self):
        self.ser.close()

    def send_left(self):
        self.ser.write(b"left\n")

    def send_right(self):
        self.ser.write(b"right\n")
    
    def send_straight(self):
        self.ser.write(b"straight\n")

    def send_stop(self):
        self.ser.write(b"stop\n")

    # no timeout, blocking
    # reads 5 bytes
    def read(self):
        return self.ser.read(5)


if __name__ == "__main__":
    print("Hi.")

