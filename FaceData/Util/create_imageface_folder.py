import cv2
import os

BaseFIlePath='H:\PyCharmWorkspace\Hannah\\'


def Filename():
    LableContent = [[0 for col in range(3)] for row in range(254)]
    annotation = open(BaseFIlePath+'resources\hannahDataset\hannah_cast.txt', "r")
    i = 0
    for eachline in annotation:
        Labs = eachline.split(('\t'), 4)
        for j in range(3):
            LableContent[i][j] = Labs[j]
        i = i + 1
    return LableContent

def mkdir(LableContent):
    for name in LableContent:
        file = "H:\PyCharmWorkspace\FaceImage\\"+name[1]
        folder = os.path.exists(file)
        if not folder:  # Determine if a folder exists and create a folder if it does not exist
            os.makedirs(file)  # function makedirs() Create this path if the path does not exist when creating the file
            print( "New folder:"+name[1])
        else:
            print("---  There is this folder!  ---")

def vedio_image():
    FrameImagePath = 'H:\PyCharmWorkspace\Hannah\\resources\Frames\\'
    vc = cv2.VideoCapture("H:\PyCharmWorkspace\Hannah\\resources\\HANNAHDataSet.avi")
    c = 1
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        rval, frame = vc.read()
        # frame=cv2.resize(frame, (996, 560), interpolation=cv2.INTER_CUBIC)
        print(FrameImagePath)
        cv2.imwrite(FrameImagePath + str(c) + '.jpg', frame)
        c = c + 1
        cv2.waitKey(1)
    vc.release()

# The main function is createing the folder of every actors


if __name__ == '__main__':
    mkdir(Filename())
    # vedio_image()