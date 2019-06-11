from firebase import firebase
import urllib.request
import os
import camscan as camscan

#Global
image_name = ''

#이미지 가져오는 함수
def get_image(URL) :
    #폴더 내에 없는 이름 찾아서 그 이름으로 저장.
    global image_name
    for i in range(0,99999):
        if not(os.path.isfile('target_image/%d.jpg'%i)):
            image_name = '%d.jpg'%i
            break  
    urllib.request.urlretrieve(URL,'target_image/%s'%image_name)

    return image_name

