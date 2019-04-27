# Get some image/annotation pairs for example 
filename_pairs = [
    # ('/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/Pellegrino_Moscato_1.jpeg',
    # '/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/SegmentationClass/IMG_0383.png'),
    # ('/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/Pellegrino_Moscato_4.jpeg',
    # '/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/SegmentationClass/IMG_0390.png'),
    # ('/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/Pellegrino_Moscato_19.jpeg',
    # '/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/SegmentationClass/IMG_0430.png')
    ('/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/Pellegrino_Moscato_1.jpeg'),
    ('/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/Pellegrino_Moscato_2.jpeg'),
    ('/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/Pellegrino_Moscato_3.jpeg')
    # ('/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/Pellegrino_Moscato_4.jpeg'),
    # ('/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/Pellegrino_Moscato_5.jpeg'),
    # ('/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/Pellegrino_Moscato_6.jpeg'),
    # ('/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/Pellegrino_Moscato_7.jpeg'),
    # ('/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/Pellegrino_Moscato_8.jpeg'),
    # ('/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/Pellegrino_Moscato_9.jpeg'),
    # ('/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/Pellegrino_Moscato_10.jpeg'),
    # ('/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/Pellegrino_Moscato_11.jpeg'),
    # ('/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/JPEGImages/Pellegrino_Moscato_12.jpeg')
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

tfrecords_filename = 'test_vino.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

# Let's collect the real images to later on compare
# to the reconstructed ones
original_images = []

# 입력받은 이미지 파일들을 img_path, annotation_path에 끝이 날 때까지 할당해준다.
for img_path in filename_pairs:

    count = 0
    # jpeg 파일을 배열로 img 변수에 집어넣는다.
    img = np.array(Image.open(img_path))
    ## png 파일을 배열로 annotation 변수에 집어넣는다.
    # annotation = np.array(Image.open(annotation_path))

    # The reason to store image sizes was demonstrated
    # in the previous example -- we have to know sizes
    # of images to later read raw serialized string,
    # convert to 1d array and convert to respective
    # shape that image used to have.
    # 1차 배열 height와 width 변수에 이미지의 크기를 집어 넣어 준다.
    height = img.shape[0]
    width = img.shape[1]

    # Put in the original images into array
    # Just for future check for correctness
    # 앞서 설정한 original_images 배열에다가 img와 annotation 이미지를 집어 넣어 준다.
    ## original_images.append((img, annotation))
    original_images.append((img))

    # 입력받은 jpeg를 binary(string) 값으로 변환
    img_raw = img.tostring()
    ## 입력받은 raw를 binary(string) 값으로 변환
    # annotation_raw = annotation.tostring()

    # tfrecord 값에 들어갈 속성 값 정의
    example = tf.train.Example(features=tf.train.Features(feature={
        'label' : _int64_feature(count),
        'height' : _int64_feature(height),
        'width' : _int64_feature(width),
        'image_raw' : _bytes_feature(img_raw)
        # 'height' : _int64_feature(height),
        # 'width' : _int64_feature(width),
        # 'image_raw' : _bytes_feature(img_raw),
        # 'mask_raw' : _bytes_feature(annotation_raw)
    }))

    count += 1
    writer.write(example.SerializeToString())

writer.close()

reconstructed_images = []

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

for string_record in record_iterator:

    example = tf.train.Example()
    example.ParseFromString(string_record)

    # tfrecord에서 height 값 만을 뽑아내서 height 변수에 할당
    height = int(example.features.feature['height'].int64_list.value[0])
    # tfrecord에서 width 값 만을 뽑아내서 height 변수에 할당
    width = int(example.features.feature['width'].int64_list.value[0])
    # tfrecord에서 label 값 만을 뽑아내서 height 변수에 할당
    label = int(example.features.feature['label'].int64_list.value[0])

    img_string = (example.features.feature['image_raw'].bytes_list.value[0])

    # annotation_string = (example.features.feature['mask_raw'].bytes_list.value[0])

    # img_string에서 추출 된 값을 string 값으로 1차 배열 형태로 img_1d에 할당
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    # img_1d 값을 (height, width, -1) 형태로 reshape
    reconstructed_img = img_1d.reshape((height, width, -1))

    # annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)

    ## Annotations don't have depth (3rd dimension)
    # reconstructed_annotation = annotation_1d.reshape((height, width, -1))

    ## reconstructed_images.append((reconstructed_img, reconstructed_annotation))
    reconstructed_images.append((reconstructed_img))

# Let's check if the reconstructed images match
# the original images

# for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):


    # img_pair_to_compare, annotation_pair_to_compare = zip(original_pair, reconstructed_pair)
    img_pair_to_compare = zip(original_pair, reconstructed_pair)

    print(np.allclose(*img_pair_to_compare))
    # print(np.allclose(*annotation_pair_to_compare))