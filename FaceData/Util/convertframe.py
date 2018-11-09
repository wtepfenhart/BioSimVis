import cv2
import io
base_path='H:\PyCharmWorkspace\Hannah'



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

def vedio_image():
    FrameImagePath = 'H:\PyCharmWorkspace\Hannah\\resources\Frames\\'
    vc = cv2.VideoCapture("H:\PyCharmWorkspace\Hannah\\resources\\1.rmvb")
    c = 1
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        rval, frame = vc.read()
        cv2.resize(frame, (996, 560), interpolation=cv2.INTER_CUBIC)
        print(FrameImagePath)
        cv2.imwrite(FrameImagePath + str(c) + '.jpg', frame)
        c = c + 1
        cv2.waitKey(1)
    vc.release()

def test():
    DateSetImageType = '.jpg'
    ImageFolder = 'H:\PyCharmWorkspace\Hannah\\resources\Frames\\'
    image = cv2.imread(ImageFolder + '2563' + '.jpg')
    x = 441
    y = 167
    w =242
    h =194
    print('H:\PyCharmWorkspace\Hannah\\resources\\test\\'+ '2822' + DateSetImageType)
    image = image[y:y+w, x:x+h]


    cv2.imwrite('H:\PyCharmWorkspace\Hannah\\resources\\test\\'+ '2822' + DateSetImageType, image)
#cut the image into the folder name by the charname
if __name__ == '__main__':
    frameid_trackid_charname=frameid_trackid_charname()
    DateSetImageType='.jpg'
    ImageFolder='H:\PyCharmWorkspace\Frames\\'
    TargetPath='H:\PyCharmWorkspace\FaceImage\\'
    i=0
    for framematrix in frameid_trackid_charname:
        frameid=str(int(framematrix[0])+68)
        if (len(frameid) < 6):
            frameid = (6 - len(frameid)) * '0' + frameid
        print('Process:',frameid)
        image=cv2.imread(ImageFolder+frameid+'.png')
        x = int(framematrix[1])
        y = int(framematrix[2])
        w = int(framematrix[3])
        h = int(framematrix[4])
        print("FilePath:",TargetPath+framematrix[5]+"\\"+frameid+DateSetImageType)
        image=image[y:y+w,x:x+h]

        cv2.imwrite(TargetPath+framematrix[5]+"\\"+frameid+DateSetImageType,image)
