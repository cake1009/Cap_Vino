import tensorflow as tf
import skimage.io as io
import matplotlib.pyplot as plt

#  File Size
IMAGE_HEIGHT = 4032
IMAGE_WIDTH = 3024

# tfrecords_filename = 'vino.tfrecords'
tfrecords_filename = 'test_vino.tfrecords'

#  tfrecord 파일을 읽어서 안에 있는 정보 풀어주는 함수
def read_and_decode(filename_queue):

    # tfrecord 파일 읽어서 reader 변수에 넣어주기
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features = {
            'image/class/label' : tf.FixedLenFeature([], tf.int64),
            'image/height' : tf.FixedLenFeature([], tf.int64),
            'image/width' : tf.FixedLenFeature([], tf.int64),
            'image/image_raw' : tf.FixedLenFeature([], tf.string),
            # 'mask_raw' : tf.FixedLenFeature([], tf.string)
        }
    )

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image/image_raw'], tf.uint8)
    ## annotation = tf.decode_raw(features['mask_raw'], tf.uint8)

    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    label = tf.cast(features['image/class/label'], tf.int32)

    image_shape = tf.stack([height, width, 3])
    ## annotation_shape = tf.stack([height, width, 1])

    image = tf.reshape(image, image_shape)
    ## annotation = tf.reshape(annotation, annotation_shape)

    image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)
    ## annotation_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=tf.int32)

    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.

    resized_image = tf.image.resize_image_with_crop_or_pad(image=image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)

    ## resized_annotation = tf.image.resize_image_with_crop_or_pad(image=image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)

    # images, annotations = tf.train.shuffle_batch( [resized_image, resized_annotation], batch_size=2, capacity=30, num_threads=2, min_after_dequeue=10)
    images = tf.train.shuffle_batch( [resized_image], batch_size=2, capacity=30, num_threads=2, min_after_dequeue=10)

    # return images, annotations
    return images

filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=10
)

# Even when reading in multiple threads, share the filename
# queue.
## image, annotation = read_and_decode(filename_queue)
image = read_and_decode(filename_queue)

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Let's read off 3 batches just for example
    for i in range(3):
        ## img, anno = sess.run([image, annotation])
        img = sess.run([image])
        print(img.__getitem__((0, :, :, :)))

        print('current batch')

        # We selected the batch size of two
        # So we should get two image pairs in each batch
        # Let's make sure it is random

        io.imshow(img[0, :, :, :])
        io.show()

        io.imshow(img[0, :, :, 0])
        io.show()

        io.imshow(img[1, :, :, :])
        io.show()

        io.imshow(img[1, :, :, 0])
        io.show()

    coord.request_stop()
    coord.join(threads)
