import camscan as camscan
import os


def find_target():
    wine = os.listdir('wineimage')
    winedir = []
    for i in wine :
        winedir.append(i)
    for i in range(len(winedir)) :
        wineimage = os.listdir('wineimage/'+winedir[i])
        for j in range(len(wineimage)):
            camscan.slice_image('wineimage/'+winedir[i]+'/'+wineimage[j],
                                'wineimage_slice/'+winedir[i]+'/'+wineimage[j] )
        
find_target()
#plt.imshow(img_binary, cmap = 'gray')
#plt.scatter([nM_x], [nM_y], c="r", s=30)
#plt.title("Middle")
#plt.show()
