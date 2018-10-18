# Hannah and Her Sister Dataset
## Introduction
Hannah Dataset is released by <a href='https://www.technicolor.com/dream/research-innovation/hannah-dataset-description'>Technicolor</a> and based on the film(Hannah and Her Sister).The whole film is divided into 153,833 frames, The size of each frames is 996×560.

Hannah and Her Sister Dataset:https://www.technicolor.com/dream/research-innovation/hannah-dataset-description




## How to Add the Label in the film.
### 1.Convert the film into the frames
```
import cv2
vc=cv2.VideoCapture(Hannah and Her Sister)
c=1
if vc.isOpened():
	rval,frame=vc.read()
else:
	rval=False
while rval:
	rval,frame=vc.read()
	cv2.waitKey(0)
	cv2.resize(frame,(996,560),interpolation=cv2.INTER_CUBIC)
	cv2.imwrite('folder'+str(c)+'.jpg',frame)
	c=c+1
	cv2.waitKey(1)
vc.release()

```
The main function of code above split the video into frames and save the frame in the form of '.jpg' name by the position of frame in the video.

### 2.Read the annotation labels
```
annotation=open('hannah_video_faces.txt',"r")

LableContent=[[0 for col in range (6)] for row in range(202178)]
i=0
for eachline in annotation:
    Labs=eachline.split(('\t'),6)
    for j in range(6):
        if j==5:
            LableContent[i][j]=Labs[5].split('\n',2)[0]
        else:
            LableContent[i][j] = Labs[j]
    i=i+1
# test code=>read the first ten data and show whether the data is right.
for i in range(10):
    print(LableContent[i])
annotation.close()

```
Tips: I skip the description and read the labels data from the sixth paragraph.
1. Create a Two-dimensional matrix
2. Add the trackid and charid in the each row.


### 3. Cast the actors' name to charid


1. Read the trackid-charid (hannah_video_tracks)

  ```
  # read the Charid  Trackid  ||  ChairId  || comment
    for eachline in CharidFile:
        Labs=eachline.split(('\t'),3)
        # print(Labs)
        Charid[i][0] = Labs[0]
        if (len(Labs)==2):
            Charid[i][1] = Labs[1].split('\n', 2)[0]
        else:
            Charid[i][1] = Labs[1]
        if (len(Labs)==3):
            Charid[i][2] = Labs[2].split('\n', 2)[0]
        i=i+1
  ```


2. Read the actorname  ChairID  || charname  ||   actorname

  ```
  n=0
  for eachline in AcotrNamefile:
      Labs=eachline.split(('\t'),3)
      Acotranme[n][0] = Labs[0]
      if (len(Labs)==2):
          Acotranme[n][1] = Labs[1].split('\n', 2)[0]
      else:
          Acotranme[n][1] = Labs[1]
      if (len(Labs)==3):
          Acotranme[n][2] = Labs[2].split('\n', 2)[0]
      n=n+1

  ```

3. Combine the Trackid with the acotrName

  ```
  for m in range(2002):
       Charidnum=m
       Acotrnamestr=Acotranme[int(Charid[Charidnum][1])-1]
       # print(Acotrnamestr)
       Charid_Acotrnane[m][0]=Charidnum+1
       if(Acotrnamestr[2]!=''):
           Charid_Acotrnane[m][1]=Acotrnamestr[1]+'    name:'+Acotrnamestr[2]
       else:
           Charid_Acotrnane[m][1] = Acotrnamestr[1]
  ```

## 4.Convert the frame into video

```
import cv2

from cv2 import VideoWriter,VideoWriter_fourcc,imread,resize

fps = 25   # video frame
fourcc = VideoWriter_fourcc(*"MJPG")
videoWriter = cv2.VideoWriter('test.avi', fourcc, fps, (996,560))  #set the video size
for i in range(2563,2653):
    img12 = cv2.imread('finished/'+str(i)+'.jpg')
#    cv2.imshow('img', img12)
#    cv2.waitKey(1000/int(fps))
    videoWriter.write(img12)
videoWriter.release()

```
