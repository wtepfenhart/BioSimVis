
import dlib
import numpy as np
import cv2
import os
import shutil

from PIL import Image

# Dlib predictor
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('dlib_dat/shape_predictor_68_face_landmarks.dat')


cap = cv2.VideoCapture(0)

# cap.set(propId, value)
cap.set(3, 480)

cnt_ss = 0

cnt_p = 0




current_face_dir = 0

path_make_dir = "data/"

path_csv = "data/"


# clear the old folders at first
def pre_clear():
    folders_rd = os.listdir(path_make_dir)
    for i in range(len(folders_rd)):
        shutil.rmtree(path_make_dir+folders_rd[i])

    csv_rd = os.listdir(path_csv)
    for i in range(len(csv_rd)):
        os.remove(path_csv+csv_rd[i])


# clear the exist folders of faces and csv
pre_clear()



person_cnt = 0

Emotion=''


while cap.isOpened():
    flag, im_rd = cap.read()
    kk = cv2.waitKey(1)
    img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)

    rects = detector(img_gray, 0)

    font = cv2.FONT_HERSHEY_COMPLEX
    # press "N" Create a file
    if kk == ord('n'):
        person_cnt += 1
        # current_face_dir = path_make_dir + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        current_face_dir = path_make_dir + "person_" + str(person_cnt)
        print('\n')
        for dirs in (os.listdir(path_make_dir)):
            if current_face_dir == path_make_dir + dirs:
                shutil.rmtree(current_face_dir)
        os.makedirs(current_face_dir)

        cnt_p = 0

    if len(rects) != 0:
        for k, d in enumerate(rects):

            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])

            height = (d.bottom() - d.top())
            width = (d.right() - d.left())

            hh = int(height/10)
            ww = int(width/8)


            cv2.rectangle(im_rd,
                          tuple([d.left()-ww, d.top()-hh]),
                          tuple([d.right()+ww, d.bottom()+hh]),
                          (0, 255, 255), 2)


            im_blank = np.zeros((height, width, 3), np.uint8)


            if kk == ord('s'):
                cnt_p += 1
                cnt_str=(str)(cnt_p)
                for ii in range(height):
                    for jj in range(width):
                        im_blank[ii][jj] = im_rd[d.top() + ii][d.left()+ jj]
                cv2.imwrite(current_face_dir + "/img_face_" + (cnt_str)+ ".jpg", im_blank)
                cv2.imwrite(current_face_dir + "/img_face_" + str(cnt_str) + ".jpg", im_blank)
                Emotion=verfiy.Verifyemotion(str(current_face_dir) + "/img_face_" + str(cnt_p) + ".jpg")


    cv2.putText(im_rd, "Emotion: " + Emotion, (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(im_rd, "N: New face folder", (20, 350), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(im_rd, "S: Recognize", (20, 400), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(im_rd, "Q: Quit", (20, 450), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    if kk == ord('q'):
        break

    cv2.namedWindow("camera", 0)
    cv2.imshow("camera", im_rd)


cap.release()

cv2.destroyAllWindows()
