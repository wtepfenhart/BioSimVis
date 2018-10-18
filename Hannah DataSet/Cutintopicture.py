
import cv2
vc=cv2.VideoCapture("1.rmvb")
c=1
if vc.isOpened():
	rval,frame=vc.read()
else:
	rval=False
while rval:
	rval,frame=vc.read()
	cv2.waitKey(0)
	cv2.resize(frame,(996,560),interpolation=cv2.INTER_CUBIC)
	cv2.imwrite('frame2/'+str(c)+'.jpg',frame)
	c=c+1
	cv2.waitKey(1)
vc.release()
