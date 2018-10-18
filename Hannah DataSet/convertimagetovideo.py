import cv2

from cv2 import VideoWriter,VideoWriter_fourcc,imread,resize

fps = 25   # video's frame rate
fourcc = VideoWriter_fourcc(*"MJPG")
videoWriter = cv2.VideoWriter('test.avi', fourcc, fps, (996,560))   #(1360,480)为视频大小
for i in range(2563,2653):
    img12 = cv2.imread('finished/'+str(i)+'.jpg')
#    cv2.imshow('img', img12)
#    cv2.waitKey(1000/int(fps))
    videoWriter.write(img12)
videoWriter.release()
