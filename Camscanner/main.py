from firebase import firebase
import urllib.request
import os
import camscan as camscan
import get_image as get_image
#import retrain as retrain


#firebase
#get() 썻을때 아무것도 없으면 None 반환
firebase = firebase.FirebaseApplication("https://cap-vino.firebaseio.com/")

image_name = ''
stop = 0
while True:
    #무한루프
    if stop == 1 :
        break
    #OUTPUT 요청 검사
    elif firebase.get('/request','URL') != None :
        #image URL 획득
        image_url = firebase.get('/request','URL')
        #URL로 부터 이미지 저장
        image_name = get_image.get_image(image_url)
        #저장된 이미지 slice
        camscan.slice_image('target_image/%s'%image_name,'target_image_slice/%s'%image_name)
        #slice된 이미지 학습된 모델로 output 구하기
        #output = retrain('target_image_slice/%s'%image_name')
        #output 파이어베이스에 저장
        #firebase.put('/request','Wine_Name',output)
        #파이어베이스에서 삭제되면 다시 요청검사
        while True:
            if firebase.get('/request','URL') == None:
                break
        image_name = ''
