import cv2
path='/Users/wangjinhu/PycharmProjects/Hannah/resources/finished/2563.jpg'

image=cv2.imread(path)
image=image[167:167+194,441:441+242]
imagepath='/Users/wangjinhu/PycharmProjects/Hannah/resources/FaceImage/LEE/test2.jpg'
cv2.imwrite(imagepath,image)