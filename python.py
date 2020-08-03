import numpy as np
import cv2
import time
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
cap = cv2.VideoCapture(0)
time.sleep(3)
for i in range(60):
  flag, background = cap.read()
background = np.flip(background,axis=1)
cv2.imwrite("background.png",background)
while(cap.isOpened()):
  flag, img = cap.read()
  if not flag:
    break
  else:
    img = np.flip(img,axis=1)
    cv2.imwrite("image.png",img)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    cv2.imwrite("hsv.png",hsv)
    lower =  np.array([0, 125, 50])
    upper = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower, upper)
    lower =  np.array([170, 170, 70])
    upper = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower, upper)
    mask1 = mask1 + mask2
    cv2.imwrite("mask1.png",mask1)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations = 20)
    mask1 = cv2.dilate(mask1, np.ones((3,3),np.uint8), iterations = 10)
    cv2.imwrite("mask1_after_morph.png",mask1)
    mask2 = cv2.bitwise_not(mask1)
    cv2.imwrite("mask2.png",mask2)
    img1 = cv2.bitwise_and(img,img,mask=mask2)
    cv2.imwrite("image_with_mask2.png",img1)
    img2 = cv2.bitwise_and(background,background,mask=mask1)
    cv2.imwrite("image_with_mask1.png",img2)
    rel = cv2.addWeighted(img1, 1, img2, 1, 0)
    cv2.imwrite("output.png",rel)
    out.write(rel)
    cv2.imshow("output",rel)
    k = cv2.waitKey(1)
    if k==27:
      break
cap.release()
out.release()
cv2.destroyAllWindows()
