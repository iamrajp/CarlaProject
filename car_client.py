import RPi.GPIO as GPIO
import time
import socket
import os
import cv2
import cPickle as pickle
from picamera import PiCamera
import sys,tty,termios
import picamera.array
import threading


IMG_SIZE=50
dmode = b'Q'
trainmode = False

host = "192.168.1.6"
port = 9004

camera = PiCamera()
s = socket.socket()
s.connect((host, port))

pin_motor00 = 11
pin_motor01 = 12
pin_motor10 = 13
pin_motor11 = 15


def setup():
    print("calling setup")
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin_motor00, GPIO.OUT)
    GPIO.setup(pin_motor01, GPIO.OUT)
    GPIO.setup(pin_motor10, GPIO.OUT)
    GPIO.setup(pin_motor11, GPIO.OUT)
    camera.rotation = 0
    camera.start_preview()
    #camera.start_recording('track_train3.h264')



def getPicture():
           
    with picamera.array.PiRGBArray(camera) as stream:
            camera.capture(stream, format='bgr')
            image = stream.array
            return image



def readImage():
    
    gray_img = cv2.cvtColor(getPicture(), cv2.COLOR_BGR2GRAY)
    img_data = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))
    img_data = pickle.dumps(img_data)
    img_data = img_data + b'<end>'
    if trainmode:
       img_data = img_data+dmode
    
    s.send(img_data)
    direction = s.recv(1024)
    return direction.decode()


def forward():
    global dmode
    dmode = b'F'
    GPIO.output(pin_motor00, 1)
    GPIO.output(pin_motor01, 0)
    GPIO.output(pin_motor10, 1)
    GPIO.output(pin_motor11, 0)





def reverse():
    global dmode
    dmode = b'Q'
    GPIO.output(pin_motor00, 0)
    GPIO.output(pin_motor01, 1)
    GPIO.output(pin_motor10,0)
    GPIO.output(pin_motor11,1)


def stop():

    global dmode
    dmode = b'Q'
    GPIO.output(pin_motor00, 0)
    GPIO.output(pin_motor01, 0)
    GPIO.output(pin_motor10, 0)
    GPIO.output(pin_motor11, 0)

def right():
    global dmode

    dmode = b'R'
    GPIO.output(pin_motor00, 1)
    GPIO.output(pin_motor01, 0)
    GPIO.output(pin_motor10, 0)
    GPIO.output(pin_motor11, 1)
    

   
def quit():

    stop()
    s.close()


def left():
    global dmode
    dmode = b'L'
   
    GPIO.output(pin_motor10, 1)
    GPIO.output(pin_motor11, 0)
    GPIO.output(pin_motor00, 0)
    GPIO.output(pin_motor01, 1)
    
   

def center():
    GPIO.output(pin_motor10, 0)
    GPIO.output(pin_motor11, 0)

setup()


class _Getch:
    def __call__(self):
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

def get():
       
        inkey = _Getch()
        k = inkey().lower()

        print("Kvalue:{0}$".format(k))

        if k=='a':
                forward()
        elif k=='b':
                reverse()
        elif k=='c':

                 right()

        elif k == 'd':

                  left()
        elif k== 'q':
                print("Quiting application")
                
                quit()
                return True
        elif k== 's':
                print("Stopping car")
                stop()
               
        elif k== 'i':
            stop()
            print("Auto mode on")
            t = threading.Thread(target=startCar)
            t.start()


        return False



def startCar():

    while True:
       
        direction = readImage()
        print(direction)
        

        if not  trainmode:

          if direction == 'F':
            
              forward()
          elif direction == 'L':
            
               left()
            
          elif direction == 'R':
          
              right()
           
                
         

         

def main():
        while True:
            if(get()):
                break



if __name__=='__main__':
    main()





GPIO.cleanup()

