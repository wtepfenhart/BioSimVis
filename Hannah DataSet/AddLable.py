import cv2
import io

def addlabs(frameid,x,y,w,h,trackid,filepath):

    image=cv2.imread('picture/'+filepath)

    # convert image size to(996,560)
    image= cv2.resize(image, (996,560), interpolation=cv2.INTER_CUBIC)
    x=int(x)
    y=int(y)
    w=int(w)
    h=int(h)

    # add lables
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 4)

    font = cv2.FONT_HERSHEY_PLAIN
    trackid=str(trackid)
    image=cv2.putText(image, trackid, (x, y), font, 2, (255, 0, 0), 2)
    filename = filepath.split('.', 2)[0]
    print(filename)
    cv2.imwrite('finished/'+filename+'.jpg',image)

#     test code
def readfile(path):
    import os
    files = os.listdir(path)  # read all the file in the folder
    s = []
    for file in files:  # read all the file  names
        s.append(file)  # add the name to the list
    return s


def readAnnotation():
    annotation = open('hannahDataset/hannah_video_faces.txt', "r")

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

def actorsname():
    CharidFile=open('hannahDataset/hannah_video_tracks.txt',"r")
    AcotrNamefile=open('hannahDataset/hannah_cast.txt',"r")
    Charid=[[0 for col in range (3)] for row in range(2002)]
    Acotranme=[[0 for col in range (3)] for row in range(254)]
    Charid_Acotrnane=[[0 for col in range (2)] for row in range(2002)]

    i = 0
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

# read the actorname  ChairID  || charname  ||   actorname
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

    for m in range(2002):
        Charidnum=m
        Acotrnamestr=Acotranme[int(Charid[Charidnum][1])-1]
        # print(Acotrnamestr)
        Charid_Acotrnane[m][0]=Charidnum+1
        if(Acotrnamestr[2]!=''):
            Charid_Acotrnane[m][1]=Acotrnamestr[1]+'    name:'+Acotrnamestr[2]
        else:
            Charid_Acotrnane[m][1] = Acotrnamestr[1]

    for i in range(2002):
        print(Charid_Acotrnane[i])
    #     close the fileflow
    CharidFile.close()
    AcotrNamefile.close()
    return  Charid_Acotrnane

if __name__ == '__main__':
    LableContent = [[0 for col in range(6)] for row in range(202178)]
    Actorname=Charid_Acotrnane=[[0 for col in range (2)] for row in range(2002)]
    Actorname=actorsname()
    LableContent=readAnnotation()
    filelist=readfile('picture')
    for i in range(len(filelist)):
        Labs=LableContent[i]
        ActorNameLable=Actorname[int(Labs[5])-1][1]
        addlabs(Labs[0],Labs[1],Labs[2],Labs[3],Labs[4],ActorNameLable,filelist[i])
