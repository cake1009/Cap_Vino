import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io

wine_img = plt.imread("/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/IMG_0383.jpeg")
plt.imshow(wine_img)
plt.show()

# cat_img = io.imread('/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/IMG_0383.jpeg')
# io.imshow(cat_img)

# Let's convert the picture into string representation
# using the ndarray.tostring() function 
# wine_string = wine_img.tostring()

# Now let's convert the string back to the image
# Important: the dtype should be specified
# otherwise the reconstruction will be errorness
# Reconstruction is 1d, so we need sizes of image
# to fully reconstruct it.
# reconstructed_wind_1d = np.fromstring(wine_string, dtype=np.uint8)

# Here we reshape the 1d representation
# This is the why we need to store the sizes of image
# along with its serialized representation.
# reconstructed_cat_img = reconstructed_wind_1d.reshape(wine_img.shape)

# Let's check if we got everything right and compare
# reconstructed array to the original one.
# print(np.allclose(wine_img, reconstructed_cat_img))