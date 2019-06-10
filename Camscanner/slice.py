import camscan as camscan
import os

#wineimage 폴더와 같은곳에 두고 실행하면
#자동으로 wineimage내의 폴더를 검색 wineimage안의 폴더별로 안에있는 이미지
#싹다 slice시켜서 wineimage_slice폴더에 똑같은 구조와 이름으로 저장시킴
def find_target():
    wine = os.listdir('wineimage')
    winedir = []
    if not(os.path.isdir('wineimage_slice')):
                os.makedirs(os.path.join('wineimage_slice'))
    for i in wine :
        winedir.append(i)
    for i in range(len(winedir)) :
        wineimage = os.listdir('wineimage/'+winedir[i])
        for j in range(len(wineimage)):
            if not(os.path.isdir('wineimage_slice/'+winedir[i])):
                os.makedirs(os.path.join('wineimage_slice/'+winedir[i]))
            camscan.slice_image('wineimage/'+winedir[i]+'/'+wineimage[j],
                                'wineimage_slice/'+winedir[i]+'/'+wineimage[j] )
        
find_target()
#plt.imshow(img_binary, cmap = 'gray')
#plt.scatter([nM_x], [nM_y], c="r", s=30)
#plt.title("Middle")
#plt.show()
