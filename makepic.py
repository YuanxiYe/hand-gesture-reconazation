import cv2
import os
import random

execution_path = os.getcwd()

def imgWrite():
        for i in range(10):
                index = 1
                for j in range(5):
                        cap = cv2.VideoCapture('videoes/%d_%d.mp4'%(i,j+1))
                        while True:
                                ret,frame = cap.read()
                                dir = None
                                if ret == False:
                                        break
                                flag = random.randint(0,999)
                                if flag < 200:
                                        dir = 'test'
                                else:
                                        dir = 'train'
                                cv2.imwrite(os.path.join(execution_path, 'gestures', dir, str(i), '4%d%d.jpg'%(flag//100, index)), frame)
                                index = index + 1
                print('1 class complete')
if __name__ == '__main__':
        imgWrite()
        print('Completed')
