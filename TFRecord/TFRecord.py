import tensorflow as tf

def bytes_feature(values):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [values]))

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    
    return tf.train.Feature(int64_list = tf.train.Int64List(value = values))

def read_imagebytes(imagefile):
    file = open(imagefile, 'rb')
    bytes = file.read()

    return bytes

def main():
    image_data = read_imagebytes('./wine_image/Golden_bubbles/Pellegrino_Moscato/IMG_0383.jpeg')
    tf_example = tf.train.Example(features = tf.train.Features(feature = {
        'image/encoded' : bytes_feature(image_data),
        'image/format' : bytes_feature(b'jpg'),
        'image/class/label' : int64_feature(1),
        'image/height' : int64_feature(192),
        'image/width' : int64_feature(256),
    }))

    writer = tf.python_io.TFRecordWriter('./output_filename.tfrecord')
    writer.write(tf_example.SerializeToString())

if __name__ == '__main__':
    main()
