import cv2

cap = cv2.VideoCapture(1)

while True:
    ret,frame = cap.read()
    if ret == False:
        break
    cv2.imshow('capture',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        cv2.imwrite('capture.jpg', frame)
        break
cv2.destroyWindow('capture')
cap.release()
