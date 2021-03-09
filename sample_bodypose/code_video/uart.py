# uart.py

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

