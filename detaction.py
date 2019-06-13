import cv2
import numpy as np
import os
from prediction import Prediction

execution_path = os.getcwd()


cap = cv2.VideoCapture(1)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('capture_detact.mp4',fourcc, 60.0, (640,480))
font = cv2.FONT_HERSHEY_SIMPLEX
flag = 1

obj_p = Prediction()
obj_p.setJsonPath(os.path.join(execution_path, "gestures", "json", "model_class.json"))
obj_p.setModelPath(os.path.join(execution_path, "gestures", "models1", "model_ex-006_acc-0.998940.h5"))
obj_p.setModelTypeAsResNet()
obj_p.loadPrediction()

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    show = frame.copy()
    frame_rgb = frame[...,::-1]
    predictions, probabilities = obj_p.predict(frame_rgb, result_count = 1,input_type = 'array')
    
    show = cv2.putText(show, predictions[0]+":"+probabilities[0], (50, 50), font, 1.2, (255, 255, 255), 2)

    cv2.namedWindow('capture1')  
    cv2.imshow('capture1',show)
    
    #out.write(show)#保存帧
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyWindow('capture1')
cap.release()
out.release()
print('Complete!')

