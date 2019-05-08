import cv2
import numpy as np
from matplotlib import pyplot as plt


def mapp(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


image=cv2.imread("4.jpg")   #read in the image
image=cv2.resize(image,(756,1000)) #resizing because opencv does not work well with bigger images
orig=image.copy()

def make_edged(image) :
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  #RGB To Gray Scale
    cv2.imshow("Title",gray)

    blurred=cv2.GaussianBlur(gray,(5,5),0)  #(5,5) is the kernel size and 0 is sigma that determines the amount of blur
    #cv2.imshow("Blur",blurred)

    edged=cv2.Canny(blurred, 30, 50)  #30 MinThreshold and 50 is the MaxThreshold
    cv2.imshow("Canny",edged)
    return edged

def find_mid(edged_img) :
    list_edged = np.where(edged_img != 0)
    middle = [sum(i) for i in list_edged]

    C_x = middle[1] / len(list_edged[1])
    C_y = middle[0] / len(list_edged[0])
    return C_x,C_y

edged = make_edged(image) #first edged

M_x,M_y = find_mid(edged) #first mid

#slice image
re_x = int(M_x) - 100
re_y = int(M_y) - 200

re_image = orig[re_y:re_y+500,re_x:re_x+200]

cv2.imwrite('org_trim.jpg',re_image)

n_edged = make_edged(re_image)

nM_x,nM_y = find_mid(n_edged)


plt.imshow(n_edged, cmap = 'gray')
plt.scatter([nM_x], [nM_y], c="r", s=30)
plt.title("Middle")
plt.show()
