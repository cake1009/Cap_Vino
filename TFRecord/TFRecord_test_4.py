import tensorflow as tf
import numpy as np
import re
from datetime import datetime

TRAINING_FILE = '/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/training_file.txt'
VALIDATION_FILE = '/Users/Myung/Desktop/0/2019_1/AI_Capston/Cap_Vino/wine_image/Golden_bubbles+Pellegrino_Moscato/validate_file.txt'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('image_width', 3024, 'help')
flags.DEFINE_integer('image_height', 4032, 'help')
flags.DEFINE_integer('image_color', 3, 'help')
flags.DEFINE_integer('maxpool_filter_size', 2, 'help')
flags.DEFINE_integer('num_classes', 5, 'help')
flags.DEFINE_integer('batch_size', 100, 'help')
flags.DEFINE_float('learning_rate', 0.0001, 'help')
flags.DEFINE_string('log_dir', '/Users/Myung/', 'help')
flags.DEFINE_integer('conv1_filter_size', 3, 'help')
flags.DEFINE_integer('conv1_layer_size', 16, 'help')
flags.DEFINE_integer('stride1', 1, 'help')
flags.DEFINE_integer('conv2_filter_size', 3, 'help')
flags.DEFINE_integer('conv2_layer_size', 32, 'help')
flags.DEFINE_integer('stride2', 1, 'help')
flags.DEFINE_integer('conv3_filter_size', 3, 'help')
flags.DEFINE_integer('conv3_layer_size', 64, 'help')
flags.DEFINE_integer('stride3', 1, 'help')
flags.DEFINE_integer('conv4_filter_size', 5, 'help')
flags.DEFINE_integer('conv4_layer_size', 128, 'help')
flags.DEFINE_integer('stride4', 1, 'help')
flags.DEFINE_integer('fc1_layer_size', 512, 'help')
flags.DEFINE_integer('fc2_layer_size', 256, 'help')

def get_input_queue(csv_file_name,num_epochs = None):
    train_images = []
    train_labels = []
    for line in open(csv_file_name,'r'):
        cols = re.split(',|\n',line)
        train_images.append(cols[0])
        # 3rd column is label and needs to be converted to int type
        train_labels.append(int(cols[2]) )
                            
    input_queue = tf.train.slice_input_producer([train_images,train_labels],num_epochs = num_epochs,shuffle = True)
    
    return input_queue

def read_data(input_queue):
    image_file = input_queue[0]
    label = input_queue[1]
    
    image =  tf.image.decode_jpeg(tf.read_file(image_file),channels=FLAGS.image_color)
    
    return image,label,image_file

def read_data_batch(csv_file_name,batch_size=FLAGS.batch_size):
    # ---
    input_queue = get_input_queue(csv_file_name)
    # 이미지 라벨 파일이름을 read_data 함수를 사용해서 각 변수에 넣어주기.
    image,label,file_name= read_data(input_queue)
    # 이미지 파일을 설정한 크기로 reshape
    image = tf.reshape(image,[FLAGS.image_height,FLAGS.image_width,FLAGS.image_color])
    
    # random image
    # 이미지를 랜덤으로 왼쪽에서 오른쪽으로 뒤집는 함수
    image = tf.image.random_flip_left_right(image)
    # 이미지의 밝기를 랜덤한 값으로 조정해 주는 함수
    image = tf.image.random_brightness(image,max_delta=0.5)
    # 이미지의 대비를 랜덤한 값으로 조정해 주는 함수
    image = tf.image.random_contrast(image,lower=0.2,upper=2.0)
    # 이미지의 RGB 영상의 색조를 랜덤 인수로 조정
    image = tf.image.random_hue(image,max_delta=0.08)
    # RGB 영상의 포화도를 랜덤 인수로 조정.
    image = tf.image.random_saturation(image,lower=0.2,upper=2.0)
    
    batch_image,batch_label,batch_file = tf.train.batch([image,label,file_name],batch_size=batch_size)
    #,enqueue_many=True)
    batch_file = tf.reshape(batch_file,[batch_size,1])

    batch_label_on_hot=tf.one_hot(tf.to_int64(batch_label),
        FLAGS.num_classes, on_value=1.0, off_value=0.0)
    return batch_image,batch_label_on_hot,batch_file

# convolutional network layer 2
def conv1(input_data):
    # layer 1 (convolutional layer)
    FLAGS.conv1_filter_size = 3
    FLAGS.conv1_layer_size = 16
    FLAGS.stride1 = 1
    
    with tf.name_scope('conv_1'):
        W_conv1 = tf.Variable(tf.truncated_normal(
                        [FLAGS.conv1_filter_size,FLAGS.conv1_filter_size,FLAGS.image_color,FLAGS.conv1_layer_size], stddev=0.1))
        b1 = tf.Variable(tf.truncated_normal(
                        [FLAGS.conv1_layer_size],stddev=0.1))
        h_conv1 = tf.nn.conv2d(input_data,W_conv1,strides=[1,1,1,1],padding='SAME')
        h_conv1_relu = tf.nn.relu(tf.add(h_conv1,b1))
        h_conv1_maxpool = tf.nn.max_pool(h_conv1_relu
                                        ,ksize=[1,2,2,1]
                                        ,strides=[1,2,2,1],padding='SAME')
        
        
    return h_conv1_maxpool

# convolutional network layer 2
def conv2(input_data):
    FLAGS.conv2_filter_size = 3
    FLAGS.conv2_layer_size = 32
    FLAGS.stride2 = 1
    
    with tf.name_scope('conv_2'):
        W_conv2 = tf.Variable(tf.truncated_normal(
                        [FLAGS.conv2_filter_size,FLAGS.conv2_filter_size,FLAGS.conv1_layer_size,FLAGS.conv2_layer_size], stddev=0.1))
        b2 = tf.Variable(tf.truncated_normal(
                        [FLAGS.conv2_layer_size],stddev=0.1))
        h_conv2 = tf.nn.conv2d(input_data,W_conv2,strides=[1,1,1,1],padding='SAME')
        h_conv2_relu = tf.nn.relu(tf.add(h_conv2,b2))
        h_conv2_maxpool = tf.nn.max_pool(h_conv2_relu
                                        ,ksize=[1,2,2,1]
                                        ,strides=[1,2,2,1],padding='SAME')
        
        
    return h_conv2_maxpool

# convolutional network layer 3
def conv3(input_data):
    FLAGS.conv3_filter_size = 3
    FLAGS.conv3_layer_size = 64
    FLAGS.stride3 = 1
    
    print ('## FLAGS.stride1 ',FLAGS.stride1)
    with tf.name_scope('conv_3'):
        W_conv3 = tf.Variable(tf.truncated_normal(
                        [FLAGS.conv3_filter_size,FLAGS.conv3_filter_size,FLAGS.conv2_layer_size,FLAGS.conv3_layer_size], stddev=0.1))
        b3 = tf.Variable(tf.truncated_normal(
                        [FLAGS.conv3_layer_size],stddev=0.1))
        h_conv3 = tf.nn.conv2d(input_data,W_conv3,strides=[1,1,1,1],padding='SAME')
        h_conv3_relu = tf.nn.relu(tf.add(h_conv3,b3))
        h_conv3_maxpool = tf.nn.max_pool(h_conv3_relu
                                        ,ksize=[1,2,2,1]
                                        ,strides=[1,2,2,1],padding='SAME')
        
        
    return h_conv3_maxpool

# convolutional network layer 3
def conv4(input_data):
    FLAGS.conv4_filter_size = 5
    FLAGS.conv4_layer_size = 128
    FLAGS.stride4 = 1
    
    with tf.name_scope('conv_4'):
        W_conv4 = tf.Variable(tf.truncated_normal(
                        [FLAGS.conv4_filter_size,FLAGS.conv4_filter_size,FLAGS.conv3_layer_size,FLAGS.conv4_layer_size], stddev=0.1))
        b4 = tf.Variable(tf.truncated_normal(
                        [FLAGS.conv4_layer_size],stddev=0.1))
        h_conv4 = tf.nn.conv2d(input_data,W_conv4,strides=[1,1,1,1],padding='SAME')
        h_conv4_relu = tf.nn.relu(tf.add(h_conv4,b4))
        h_conv4_maxpool = tf.nn.max_pool(h_conv4_relu
                                        ,ksize=[1,2,2,1]
                                        ,strides=[1,2,2,1],padding='SAME')
        
        
    return h_conv4_maxpool

# fully connected layer 1
def fc1(input_data):
    input_layer_size = 6*6*FLAGS.conv4_layer_size
    FLAGS.fc1_layer_size = 512
    
    with tf.name_scope('fc_1'):
        # 앞에서 입력받은 다차원 텐서를 fcc에 넣기 위해서 1차원으로 피는 작업
        input_data_reshape = tf.reshape(input_data, [-1, input_layer_size])
        W_fc1 = tf.Variable(tf.truncated_normal([input_layer_size,FLAGS.fc1_layer_size],stddev=0.1))
        b_fc1 = tf.Variable(tf.truncated_normal(
                        [FLAGS.fc1_layer_size],stddev=0.1))
        h_fc1 = tf.add(tf.matmul(input_data_reshape,W_fc1) , b_fc1) # h_fc1 = input_data*W_fc1 + b_fc1
        h_fc1_relu = tf.nn.relu(h_fc1)
    
    return h_fc1_relu
    
# fully connected layer 2
def fc2(input_data):
    FLAGS.fc2_layer_size = 256
    
    with tf.name_scope('fc_2'):
        W_fc2 = tf.Variable(tf.truncated_normal([FLAGS.fc1_layer_size,FLAGS.fc2_layer_size],stddev=0.1))
        b_fc2 = tf.Variable(tf.truncated_normal(
                        [FLAGS.fc2_layer_size],stddev=0.1))
        h_fc2 = tf.add(tf.matmul(input_data,W_fc2) , b_fc2) # h_fc1 = input_data*W_fc1 + b_fc1
        h_fc2_relu = tf.nn.relu(h_fc2)
    
    return h_fc2_relu

# final layer
def final_out(input_data):

    with tf.name_scope('final_out'):
        W_fo = tf.Variable(tf.truncated_normal([FLAGS.fc2_layer_size,FLAGS.num_classes],stddev=0.1))
        b_fo = tf.Variable(tf.truncated_normal(
                        [FLAGS.num_classes],stddev=0.1))
        h_fo = tf.add(tf.matmul(input_data,W_fo) , b_fo) # h_fc1 = input_data*W_fc1 + b_fc1
        
    # 최종 레이어에 softmax 함수는 적용하지 않았다. 
        
    return h_fo

# build cnn_graph
def build_model(images,keep_prob):
    # define CNN network graph
    # output shape will be (*,48,48,16)
    r_cnn1 = conv1(images) # convolutional layer 1
    print ("shape after cnn1 ",r_cnn1.get_shape())
    
    # output shape will be (*,24,24,32)
    r_cnn2 = conv2(r_cnn1) # convolutional layer 2
    print ("shape after cnn2 :",r_cnn2.get_shape() )
    
    # output shape will be (*,12,12,64)
    r_cnn3 = conv3(r_cnn2) # convolutional layer 3
    print ("shape after cnn3 :",r_cnn3.get_shape() )

    # output shape will be (*,6,6,128)
    r_cnn4 = conv4(r_cnn3) # convolutional layer 4
    print ("shape after cnn4 :",r_cnn4.get_shape() )
    
    # fully connected layer 1
    r_fc1 = fc1(r_cnn4)
    print ("shape after fc1 :",r_fc1.get_shape() )

    # fully connected layer2
    r_fc2 = fc2(r_fc1)
    print ("shape after fc2 :",r_fc2.get_shape() )
    
    ## drop out
    # 참고 http://stackoverflow.com/questions/34597316/why-input-is-scaled-in-tf-nn-dropout-in-tensorflow
    # 트레이닝시에는 keep_prob < 1.0 , Test 시에는 1.0으로 한다. 
    r_dropout = tf.nn.dropout(r_fc2,keep_prob)
    print ("shape after dropout :",r_dropout.get_shape() ) 
    
    # final layer
    r_out = final_out(r_dropout)
    print ("shape after final layer :",r_out.get_shape() )


    return r_out 

def main(argv=None):
    
    # define placeholders for image data & label for traning dataset
    
    images = tf.placeholder(tf.float32,[None,FLAGS.image_height,FLAGS.image_width,FLAGS.image_color])
    labels = tf.placeholder(tf.int32,[None,FLAGS.num_classes])
    image_batch,label_batch,file_batch = read_data_batch(TRAINING_FILE) 

    keep_prob = tf.placeholder(tf.float32) # dropout ratio
    prediction = build_model(images,keep_prob)
    # define loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=labels))

    tf.summary.scalar('loss',loss)

    #define optimizer
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    train = optimizer.minimize(loss)


    # for validation
    #with tf.name_scope("prediction"):
    validate_image_batch,validate_label_batch,validate_file_batch = read_data_batch(VALIDATION_FILE)
    label_max = tf.argmax(labels,1)
    pre_max = tf.argmax(prediction,1)
    correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
            
    tf.summary.scalar('accuracy',accuracy)
        
    
    startTime = datetime.now()
    
    #build the summary tensor based on the tF collection of Summaries
    summary = tf.summary.merge_all()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        saver = tf.train.Saver() # create saver to store training model into file
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir,sess.graph)
        
        init_op = tf.global_variables_initializer() # use this for tensorflow 0.12rc0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(init_op)
        
        for i in range(10000):
            images_,labels_ = sess.run([image_batch,label_batch])
            #sess.run(train_step,feed_dict={images:images_,labels:labels_,keep_prob:0.8})
            sess.run(train,feed_dict={images:images_,labels:labels_,keep_prob:0.7})
            
            if i % 10 == 0:
                now = datetime.now()-startTime
                print('## time:',now,' steps:',i)         
                
                # print out training status
                rt = sess.run([label_max,pre_max,loss,accuracy],feed_dict={images:images_ , labels:labels_, keep_prob:1.0})
                print ('Prediction loss:',rt[2],' accuracy:',rt[3])
                # validation steps
                validate_images_,validate_labels_ = sess.run([validate_image_batch,validate_label_batch])
                rv = sess.run([label_max,pre_max,loss,accuracy],feed_dict={images:validate_images_ , labels:validate_labels_, keep_prob:1.0})
                print ('Validation loss:',rv[2],' accuracy:',rv[3])
                if(rv[3] > 0.9):
                    break
                # validation accuracy
                summary_str = sess.run(summary,feed_dict={images:validate_images_ , labels:validate_labels_, keep_prob:1.0})
                summary_writer.add_summary(summary_str,i)
                summary_writer.flush()
        
        saver.save(sess, 'face_recog') # save session
        coord.request_stop()
        coord.join(threads)
        print('finish')
    
main()
