# Build Face image data on hannah
This data all cut from the film hannah and her sister.
We try to extract the image of faces and named them by the frameid

 ## STATISTICS
 [Technicolor Introduction](https://www.technicolor.com/dream/research-innovation/hannah-dataset-description):
- 153,833 frames, size 996×560
- 245 shots
- 202,178 bounding faces boxes
- 2,002 face tracks
- 1,518 speech segments
- 400 audio segments with non-speech human sounds (laughing, screaming, kissing, etc.)
- 254 labels: 53 named characters, 186 identified un-named characters, 15 crowds

## Process：
1. Convert from film to frames
2. Create actors' Face folder
3. Extract face image

  ### Convert from film to frames
    1. Python

  ```
  import cv2
      vc=cv2.VideoCapture("vedionamepath")
      c=1
      if vc.isOpened():
      	rval,frame=vc.read()
      else:
      	rval=False
      while rval:
      	rval,frame=vc.read()
      	cv2.waitKey(0)
      	cv2.resize(frame,(996,560),interpolation=cv2.INTER_CUBIC)
      	cv2.imwrite('framepath'+str(c)+'.jpg',frame)
      	c=c+1
      	cv2.waitKey(1)
      vc.release()
  ```
  ![Image](https://github.com/wtepfenhart/BioSimVis/blob/master/ExperientImage/FrameImage.png)
      2.    ffmpeg [Install]()

  ```
  ffmpeg -i input.mov -r 25 output_%04d.png
  ```
  Tips:  Every second 25 pictures

  ### Create actors's face folder

  ```
    BaseFIlePath='/Users/wangjinhu/PycharmProjects/Hannah/'
    def Filename():
        LableContent = [[0 for col in range(3)] for row in range(254)]
        annotation = open(BaseFIlePath+'resources/hannahDataset/hannah_cast.txt', "r")
        i = 0
        for eachline in annotation:
            Labs = eachline.split(('\t'), 4)
            for j in range(3):
                LableContent[i][j] = Labs[j]
            i = i + 1
        return LableContent

    def mkdir(LableContent):
        for name in LableContent:
            file = BaseFIlePath+"resources/FaceImage/"+name[1]
            folder = os.path.exists(file)
            if not folder:  # Determine if a folder exists and create a folder if it does not exist
                os.makedirs(file)  # function makedirs() Create this path if the path does not exist when creating the file
                print( "New folder:"+name[1])
            else:
                print("---  There is this folder!  ---")
    # The main function is createing the folder of every actors
  ```

  ### Extract face image
  1. Read the dataset's name Labs
    * Target:
      Save the face image in the folder and named by the frameid
  2. Make the cast table :
  ![image](https://github.com/wtepfenhart/BioSimVis/blob/master/ExperientImage/face_extract_cast.jpg)
        frameid | x | Y | w | H | acotrname

  ```
    # return matrix frameid  | x | Y | W | H | Trackid
    def readAnnotation():

        annotation = open(base_path+'\\resources\hannahDataset\hannah_video_faces.txt', "r")

        LableContent = [[0 for col in range(6)] for row in range(202178)]
        i = 0
        for eachline in annotation:
            Labs = eachline.split(('\t'), 6)
            for j in range(6):
                if j == 5:
                    LableContent[i][j] = Labs[5].split('\n', 2)[0]
                else:
                    LableContent[i][j] = Labs[j]
            i = i + 1
        annotation.close()
        return LableContent

    def frameid_trackid_charname():
        CharidFile=open(base_path+'\\resources\hannahDataset\hannah_video_tracks.txt',"r")
        AcotrNamefile=open(base_path+'\\resources\hannahDataset\hannah_cast.txt',"r")
        Charid=[[0 for col in range (3)] for row in range(2002)]
        Acotranme=[[0 for col in range (3)] for row in range(254)]
        frameid_coordinate_charname=[[0 for col in range (6)] for row in range(202178)]
        i = 0

    # read the Labs() Charid  Trackid  ||  ChairId  || comment
        for eachline in CharidFile:
            Labs=eachline.split(('\t'),3)
            Charid[i][0] = Labs[0]
            if (len(Labs)==2):
                Charid[i][1] = Labs[1].split('\n', 2)[0]
            else:
                Charid[i][1] = Labs[1]
            if (len(Labs)==3):
                Charid[i][2] = Labs[2].split('\n', 2)[0]
            i=i+1

    # read  actorname  ChairID  || charname  ||   actorname
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
    # read  frameid || coordinate || actorname
        frameid_coordinate = readAnnotation()
        for i in range(202178):
            for j in range(5):
                frameid_coordinate_charname[i][j]=frameid_coordinate[i][j]
            trackid=int(frameid_coordinate[i][5])
            charid=int(Charid[trackid-1][1])
            actorname=Acotranme[charid-1][1]
            frameid_coordinate_charname[i][5]=actorname

        return frameid_coordinate_charname

  ```

  ## Result
![Image](https://github.com/wtepfenhart/BioSimVis/blob/master/ExperientImage/FaceImage.png)
