# Get some image/annotation pairs for example 
filename_pairs = [
    ('/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/IMG_0383.jpeg',
    '/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/SegmentationClass/IMG_0383.png')
]

# Important: We are using PIL to read .png files later.
# This was done on purpose to read indexed png files
# in a special way -- only indexes and not map the indexes
# to actual rgb values. This is specific to PASCAL VOC
# dataset data. If you don't want thit type of behaviour
# consider using skimage.io.imread()

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage.io as io

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tfrecords_filename = 'pascal_voc_segmentation.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

# Let's collect the real images to later on compare
# to the reconstructed ones
original_images = []

for img_path, annotation_path in filename_pairs:

    img = np.array(Image.open(img_path))
    annotation = np.array(Image.open(annotation_path))

    # The reason to store image sizes was demonstrated
    # in the previous example -- we have to know sizes
    # of images to later read raw serialized string,
    # convert to 1d array and convert to respective
    # shape that image used to have.
    height = img.shape[0]
    width = img.shape[1]

    # Put in the original images into array
    # Just for future check for correctness
    original_images.append((img, annotation))

    img_raw = img.tostring()
    annotation_raw = annotation.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height' : _int64_feature(height),
        'width' : _int64_feature(width),
        'image_raw' : _bytes_feature(img_raw),
        'mask_raw' : _bytes_feature(annotation_raw)
    }))

    writer.write(example.SerializeToString())

writer.close()

reconstructed_images = []

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

for string_record in record_iterator:

    example = tf.train.Example()
    example.ParseFromString(string_record)

    height = int(example.features.feature['height'].int64_list.value[0])

    width = int(example.features.feature['width'].int64_list.value[0])

    img_string = (example.features.feature['image_raw'].bytes_list.value[0])

    annotation_string = (example.features.feature['mask_raw'].bytes_list.value[0])

    img_1d = np.fromstring(img_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((height, width, -1))

    annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)

    # Annotations don't have depth (3rd dimension)
    reconstructed_annotation = annotation_1d.reshape((height, width, -1))

    reconstructed_images.append((reconstructed_img, reconstructed_annotation))

# Let's check if the reconstructed images match
# the original images

for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):

    img_pair_to_compare, annotation_pair_to_compare = zip(original_pair, reconstructed_pair)

    print(np.allclose(*img_pair_to_compare))
    print(np.allclose(*annotation_pair_to_compare))