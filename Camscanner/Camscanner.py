import cv2
import numpy as np
import time
from matplotlib import pyplot as plt


print("import ok")

def slice_image(target_image,stored_dir):
    
    image=cv2.imread(target_image)
    image=cv2.resize(image,(756,1000)) #resizing because opencv does not work well with bigger images
    orig=image.copy()
    #cv2.imshow("Title",image)
    
    def make_edged(image) :
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  #RGB To Gray Scale
        #cv2.imshow("Title",gray)
        
    
        blurred=cv2.GaussianBlur(gray,(5,5),0)  #(5,5) is the kernel size and 0 is sigma that determines the amount of blur
        #cv2.imshow("Blur",blurred)

        edged=cv2.Canny(blurred, 50, 70)  #30 MinThreshold and 50 is the MaxThreshold
        #cv2.imshow("Canny",edged)

        return edged


    def find_mid(edged_img) :
        #가중치400~850 10배
        weight = 10
        list_edged = np.where(edged_img != 0)
        list_edged = list(list_edged)
        list_edged[0] = list(list_edged[0])
        list_edged[1] = list(list_edged[1])
        for i in range(len(list_edged[0])):
            if 400<list_edged[0][i]<850 :
                for j in range(weight) :
                    list_edged[1].append(list_edged[1][i])
                    list_edged[0].append(list_edged[0][i])
        for i in range(len(list_edged[1])) :
            if 250<list_edged[1][i] < 500 :
                for j in range(weight) :
                    list_edged[1].append(list_edged[1][i])
                    list_edged[0].append(list_edged[0][i])
                
        middle = [sum(i) for i in list_edged]

        C_x = middle[1] / len(list_edged[1])
        C_y = middle[0] / len(list_edged[0])
        #print(C_x,C_y)
        return C_x,C_y

    edged = make_edged(image) #first edged
    #cv2.imshow("aa",edged)

    M_x,M_y = find_mid(edged) #first mid

    #slice image
    re_x = int(M_x) - 120
    re_y = int(M_y) - 400
    #print(re_x,re_y)
    re_image = orig[re_y:re_y+800,re_x:re_x+240]
    re_image = cv2.resize(re_image,(300,400))
    slice_image = 'slice_' + target_image
    cv2.imwrite(stored_dir,re_image)
    #print(slice_image + ' compleate')
    n_edged = make_edged(re_image)

    #plt.imshow(edged, cmap = 'gray')
    #plt.scatter([M_x], [M_y], c="r", s=30)
    #plt.title("Middle")
    #plt.show()

