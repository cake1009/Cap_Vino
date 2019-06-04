from datetime import datetime
import hashlib
import os
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.contrib.quantize.python import quant_ops
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

# module level variables ##############################################################################################
# 학습하는데 최소 학습 데이터 수
MIN_NUM_IMAGES_REQUIRED_FOR_TRAINING = 10
# 학습하는데 권장하는 학습 데이터 수
MIN_NUM_IMAGES_SUGGESTED_FOR_TRAINING = 100
# 테스트하는데 최소 테스트 데이터 수
MIN_NUM_IMAGES_REQUIRED_FOR_TESTING = 3
# 각 샘플 별 최대 이미지 파일 크기 : ~134M
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1

# path to folders of labeled images
# os.getcwd() : 현재 사용하고 있는 프로세스의 dictionary 위치를 unicode string으로 알려준다.
# 학습 데이터 위치
TRAINING_IMAGES_DIR = os.getcwd() + '/training_images'
# 테스트 데이터 위치
TEST_IMAGES_DIR = os.getcwd() + "/test_images/"

# where to save the trained graph
# 학습하고 저장된 graph file
OUTPUT_GRAPH = os.getcwd() + '/' + 'retrained_graph.pb'

# where to save the intermediate graphs
# ****** 중간 그래프 저장 위치
INTERMEDIATE_OUTPUT_GRAPHS_DIR = os.getcwd() + '/intermediate_graph'

# how many steps to store intermediate graph, if "0" then will not store
# ****** 중간 그래프가 얼마나 보관되어 있는지 확인하는 함수 - 0일 경우 저장 안됨
INTERMEDIATE_STORE_FREQUENCY = 0

# where to save the trained graph's labels
# 학습된 그래프 라벨 저장 위치
OUTPUT_LABELS = os.getcwd() + '/' + 'retrained_labels.txt'

# where to save summary logs for TensorBoard
# 학습의 진행사항을 볼 수 있는 텐서보드 저장 위치
TENSORBOARD_DIR = os.getcwd() + '/' + 'tensorboard_logs'

# how many training steps to run before ending
# NOTE: original Google default is 4000, use 4000 (or possibly higher) for production grade results
# 학습을 몇번 동안 할 것인지 지정 - 구글은 디폴트 값으로 4000을 권장한다.
HOW_MANY_TRAINING_STEPS=300

# how large a learning rate to use when training
# learning rate 값 지정
LEARNING_RATE = 0.001

# what percentage of images to use as a test set
# 테스트 셋에 비례하여 사용할 이미지 percentage
TESTING_PERCENTAGE = 10

# what percentage of images to use as a validation set
# 발리데이션 셋에 비례하여 사용할 이미지 percentage
VALIDATION_PERCENTAGE = 10

# how often to evaluate the training results
# 학습 결과를 몇번 평가할 것인가.
EVAL_STEP_INTERVAL = 10

# how many images to train on at a time
# 학습할 때 사용할 배치 사이츠 지정
TRAIN_BATCH_SIZE = 100

# How many images to test on. This test set is only used once, to evaluate the final accuracy of the model after
# training completes.  A value of -1 causes the entire test set to be used, which leads to more stable results across runs.
# 테스트 때 사용할 배치 사이즈 지정 - '-1'이 의미하는 것은 가지고 있는 테스트 이미지를 하나하나 다 사용하겠다는 것을 의미.
TEST_BATCH_SIZE = -1

# How many images to use in an evaluation batch. This validation set is used much more often than the test set, and is an early indicator of how
# accurate the model is during training. A value of -1 causes the entire validation set to be used, which leads to
# more stable results across training iterations, but may be slower on large training sets.
# 발리데이션 때 사용할 배치 사이즈 지정.
VALIDATION_BATCH_SIZE = 100

# 분류하지 못한 학습 데이터를 나타내는 값
PRINT_MISCLASSIFIED_TEST_IMAGES = False

# 여러 데이터 파일들의 경로 설정
MODEL_DIR = os.getcwd() + "/" + "model"

# 이미지 파일을 그대로 사용할 경우 과한 버퍼링을 초래할 수 있으므로 이를 해결하기 위해 .txt형태로 이미지 파일 변환
BOTTLENECK_DIR = os.getcwd() + '/' + 'bottleneck_data'

# 마지막 학습된 텐서 그래프 이름 설정
FINAL_TENSOR_NAME = 'final_result'

# 수평 학습된 이미지를 랜덤으로 플립 할 것인가
FLIP_LEFT_RIGHT = False

# 학습 이미지를 랜덤하게 잘라낼 여백을 결정하는 백분율
RANDOM_CROP = 0

# 학습 이미지 크기를 랜덤하게 키울 비율을 결정하는 백분율
RANDOM_SCALE = 0

# 학습 이미지 입력 픽셀을 위 또는 아래로 무작위로 곱하여 비율을 결정하는 백분율
RANDOM_BRIGHTNESS = 0

# 사용할 모델 아키텍처. '인셉션_v3'는 높은 정확성을 가지고 있으나 느리다는 단점을 가진다. 더 빠르거나 느린 모델인 경우, 'mobilenet_<parameter size>_<input_size>[_quantized]' 
# 이 형식의 MobileNet을 선택하는 것이 좋다. 예를 들어 'mobilenet_1.0_224'는 크기가 17MB인 모델을 선택하고 224개 픽셀 입력 이미지를 받으며, 'mobilenet_0.25_128_quantized'는 
# 훨씬 정확도는 낮지만 디스크에 920KB, 128x128 이미지 크기를 가진 작고 빠른 네트워크를 선택할 것이다.
# https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html 참고

# 높은 정확도를 사용하지만 상대적으로 크고 느린 Inception v3 모델 아키텍처를 사용한다.
# 좋은 교육 데이터를 수집했는지를 검증하기 위해 이 작업부터 시작하는 것이 좋으나, 연구 제한 플랫폼에 배치하려면 MobildNet 모델로 "--architecture" 플래그를 사용할 수 있다.

# 플로팅 포인트 버전의 moupenet을 실행하는 명령 예:
# ARCHITECTURE = 'mobilenet_1.0_224'

# 정량화된 버전의 moupenet을 실행하는 명령 예:
# ARCHITECTURE = 'mobilenet_1.0_224_quantized'

# 선택할 수 있는 32가지 모빌레넷 모델이 있으며, 다양한 파일 크기와 지연 시간 옵션이 있다. 
# 첫 번째 숫자는 크기를 제어하기 위해 '1.0', '0.75', '0.50' 또는 '0.25'가 될 수 있으며, 
# 두 번째 숫자는 입력 이미지 크기를 '224', '192', '160' 또는 '128' 중 하나로 제어하며 작은 크기가 더 빠르게 실행된다.
ARCHITECTURE = 'inception_v3'


#######################################################################################################################
def main():
    print("프로그램 시작...")

    tf.logging.set_verbosity(tf.logging.INFO)

    # 학습 디렉토리가 없을 때 에러 표시문:
    if not checkIfNecessaryPathsAndFilesExist():
        return

    # 학습 중에 사용할 수 있는 필수 디렉토리 준비.
    prepare_file_system()

    # 사용할 모델 아키텍처(inception_v3)에 대한 정보 확인하기.
    model_info = create_model_info(ARCHITECTURE)
    if not model_info:
        tf.logging.error('아키텍쳐를 확인할 수 없습니다.')
        return -1

    # 모델이 없으면 다운로드한 다음 모델 그래프를 생성
    print("모델 다운로드...")
    downloadModelIfNotAlreadyPresent(model_info['data_url'])
    print("모델 그래프 생성...")
    graph, bottleneck_tensor, resized_image_tensor = (create_model_graph(model_info))

    # Look at the folder structure, and create lists of all the images.
    print("creating image lists . . .")
    image_lists = create_image_lists(TRAINING_IMAGES_DIR, TESTING_PERCENTAGE, VALIDATION_PERCENTAGE)
    class_count = len(image_lists.keys())
    if class_count == 0:
        tf.logging.error('No valid folders of images found at ' + TRAINING_IMAGES_DIR)
        return -1
    # end if
    if class_count == 1:
        tf.logging.error('Only one valid folder of images found at ' + TRAINING_IMAGES_DIR + ' - multiple classes are needed for classification.')
        return -1
    # end if

    # determinf if any of the distortion command line flags have been set
    doDistortImages = False
    if (FLIP_LEFT_RIGHT == True or RANDOM_CROP != 0 or RANDOM_SCALE != 0 or RANDOM_BRIGHTNESS != 0):
        doDistortImages = True
    # end if

    print("starting session . . .")
    with tf.Session(graph=graph) as sess:
        # Set up the image decoding sub-graph.
        print("performing jpeg decoding . . .")
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding( model_info['input_width'],
                                                                    model_info['input_height'],
                                                                    model_info['input_depth'],
                                                                    model_info['input_mean'],
                                                                    model_info['input_std'])
        print("caching bottlenecks . . .")
        distorted_jpeg_data_tensor = None
        distorted_image_tensor = None
        if doDistortImages:
            # We will be applying distortions, so setup the operations we'll need.
            (distorted_jpeg_data_tensor, distorted_image_tensor) = add_input_distortions(FLIP_LEFT_RIGHT, RANDOM_CROP, RANDOM_SCALE,
                                                                                         RANDOM_BRIGHTNESS, model_info['input_width'],
                                                                                         model_info['input_height'], model_info['input_depth'],
                                                                                         model_info['input_mean'], model_info['input_std'])
        else:
            # We'll make sure we've calculated the 'bottleneck' image summaries and
            # cached them on disk.
            cache_bottlenecks(sess, image_lists, TRAINING_IMAGES_DIR, BOTTLENECK_DIR, jpeg_data_tensor, decoded_image_tensor,
                              resized_image_tensor, bottleneck_tensor, ARCHITECTURE)
        # end if

        # Add the new layer that we'll be training.
        print("adding final training layer . . .")
        (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = add_final_training_ops(len(image_lists.keys()),
                                                                                                                 FINAL_TENSOR_NAME,
                                                                                                                 bottleneck_tensor,
                                                                                                                 model_info['bottleneck_tensor_size'],
                                                                                                                 model_info['quantize_layer'])
        # Create the operations we need to evaluate the accuracy of our new layer.
        print("adding eval ops for final training layer . . .")
        evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)

        # Merge all the summaries and write them out to the tensorboard_dir
        print("writing TensorBoard info . . .")
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(TENSORBOARD_DIR + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(TENSORBOARD_DIR + '/validation')

        # Set up all our weights to their initial default values.
        init = tf.global_variables_initializer()
        sess.run(init)

        # Run the training for as many cycles as requested on the command line.
        print("performing training . . .")
        for i in range(HOW_MANY_TRAINING_STEPS):
            # Get a batch of input bottleneck values, either calculated fresh every
            # time with distortions applied, or from the cache stored on disk.
            if doDistortImages:
                (train_bottlenecks, train_ground_truth) = get_random_distorted_bottlenecks(sess, image_lists, TRAIN_BATCH_SIZE, 'training',
                                                                                           TRAINING_IMAGES_DIR, distorted_jpeg_data_tensor,
                                                                                           distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
            else:
                (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(sess, image_lists, TRAIN_BATCH_SIZE, 'training',
                                                                                           BOTTLENECK_DIR, TRAINING_IMAGES_DIR, jpeg_data_tensor,
                                                                                           decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                                                                                           ARCHITECTURE)
            # end if

            # Feed the bottlenecks and ground truth into the graph, and run a training
            # step. Capture training summaries for TensorBoard with the `merged` op.
            train_summary, _ = sess.run([merged, train_step], feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)

            # 잘 학습되고 있는지 출력.
            is_last_step = (i + 1 == HOW_MANY_TRAINING_STEPS)
            if (i % EVAL_STEP_INTERVAL) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy], feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
                tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i, train_accuracy * 100))
                tf.logging.info('%s: Step %d: Cross entropy = %f' % (datetime.now(), i, cross_entropy_value))
                validation_bottlenecks, validation_ground_truth, _ = (get_random_cached_bottlenecks(sess, image_lists, VALIDATION_BATCH_SIZE, 'validation',
                                                                                                    BOTTLENECK_DIR, TRAINING_IMAGES_DIR, jpeg_data_tensor,
                                                                                                    decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                                                                                                    ARCHITECTURE))
                # validation를 하고 Merged op으로 TensorBoard의 학습 상태 저장.
                validation_summary, validation_accuracy = sess.run(
                    [merged, evaluation_step], feed_dict={bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                validation_writer.add_summary(validation_summary, i)
                tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' % (datetime.now(), i, validation_accuracy * 100, len(validation_bottlenecks)))
            # end if

            # Store intermediate results
            intermediate_frequency = INTERMEDIATE_STORE_FREQUENCY

            if (intermediate_frequency > 0 and (i % intermediate_frequency == 0) and i > 0):
                intermediate_file_name = (INTERMEDIATE_OUTPUT_GRAPHS_DIR + 'intermediate_' + str(i) + '.pb')
                tf.logging.info('Save intermediate result to : ' + intermediate_file_name)
                save_graph_to_file(sess, graph, intermediate_file_name)
            # end if
        # end for

        # We've completed all our training, so run a final test evaluation on some new images we haven't used before
        print("running testing . . .")
        test_bottlenecks, test_ground_truth, test_filenames = (get_random_cached_bottlenecks(sess, image_lists, TEST_BATCH_SIZE, 'testing', BOTTLENECK_DIR,
                                                                                             TRAINING_IMAGES_DIR, jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                                                                                             bottleneck_tensor, ARCHITECTURE))
        test_accuracy, predictions = sess.run([evaluation_step, prediction], feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))

        if PRINT_MISCLASSIFIED_TEST_IMAGES:
            tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
            for i, test_filename in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i]:
                    tf.logging.info('%70s  %s' % (test_filename, list(image_lists.keys())[predictions[i]]))
                # end if
            # end for
        # end if

        # write out the trained graph and labels with the weights stored as constants
        print("writing trained graph and labbels with weights")
        save_graph_to_file(sess, graph, OUTPUT_GRAPH)
        with gfile.FastGFile(OUTPUT_LABELS, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')
        # end with

        print("done !!")
# end function

#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    # 학습 디렉토리가 없는 경우, 에러 메세지 표시
    if not os.path.exists(TRAINING_IMAGES_DIR):
        print('')
        print('에러: TRAINING_IMAGES_DIR "' + TRAINING_IMAGES_DIR + '" 위치에 존재하지 않는다')
        print('학습 이미지를 올바르게 넣어 놓을것')
        print('')
        return False

    class TrainingSubDir:
        def __init__(self):
            self.loc = ""
            self.numImages = 0

    # 학습 서브 디렉토리 리스트 위치 
    trainingSubDirs = []

    # 학습 서브 디렉토리를 .txt파일 형식으로 만들기 위한 준비
    # os.listdir() : OS 모듈에 속하는 os.listdir 함수를 사용하여 ".txt"로 끝나는 모든 파일 특정 경로(".") 검색한다.
    for dirName in os.listdir(TRAINING_IMAGES_DIR):
        currentTrainingImagesSubDir = os.path.join(TRAINING_IMAGES_DIR, dirName)
        # os.path.isdir(): 경로가 기존 디렉터리인 경우 True 리턴. 
        if os.path.isdir(currentTrainingImagesSubDir):
            trainingSubDir = TrainingSubDir()
            trainingSubDir.loc = currentTrainingImagesSubDir
            trainingSubDirs.append(trainingSubDir)

    # 학습 서브 디렉터리가 보이지 않으면 오류 메시지를 보여주고 false 리턴
    if len(trainingSubDirs) == 0:
        print("에러: " + TRAINING_IMAGES_DIR + "에 학습시킬 분류된 서브 이미지 파일이 보이지 않습니다.")
        print("학습할 데이터 이미지 파일을 구분할 종류에 맞게 끔 파일을 분리하여 만드세요.")
        return False

    # 각 학습 서브 디렉토리에 교육 이미지들을 채우기.
    for trainingSubDir in trainingSubDirs:
        # 현재 서브 디렉토리에 이미지가 몇개 있는지 확인하기
        for fileName in os.listdir(trainingSubDir.loc):
            if fileName.endswith(".jpg"):
                trainingSubDir.numImages += 1

    # 학습 서브 디렉토리에 필요한 최소 학습 이미지 수보다 부족하면 오류 메시지를 보여주고 false 리턴
    for trainingSubDir in trainingSubDirs:
        if trainingSubDir.numImages < MIN_NUM_IMAGES_REQUIRED_FOR_TRAINING:
            print("에러: 최소 학습 이미지 데이터 수인 " + str(MIN_NUM_IMAGES_REQUIRED_FOR_TRAINING) + " 개가 필요합니다. - " + trainingSubDir.loc)
            print("학습 이미지가 부족하지 않게 충분히 채워 넣으세요.")
            return False

    # 학습 서브 디렉토리가 권장한 학습 이미지 수보다 작을 경우 경고 표시
    for trainingSubDir in trainingSubDirs:
        if trainingSubDir.numImages < MIN_NUM_IMAGES_SUGGESTED_FOR_TRAINING:
            print("주의: 적정 학습 이미지 데이터 수인 " + str(MIN_NUM_IMAGES_SUGGESTED_FOR_TRAINING) + " 개가 필요합니다. - " + trainingSubDir.loc)
            print("더 좋은 결과를 학습하기 위해 이미지 추가가 필요로 합니다.")

    # 테스트 디렉토리에 이미지가 존재하지 않을 경우, 에러 메세지를 보여주고 false 리턴
    if not os.path.exists(TEST_IMAGES_DIR):
        print('')
        print('에러: TEST_IMAGES_DIR "' + TEST_IMAGES_DIR + '" 에 테스트 할 이미지 파일이 보이지 않습니다.')
        print('테스트 이미지를 확인해 주세요.')
        print('')
        return False

    # 테스트 이미지 디렉토리에 있는 이미지 수 카운트
    numImagesInTestDir = 0
    for fileName in os.listdir(TEST_IMAGES_DIR):
        if fileName.endswith(".jpg"):
            numImagesInTestDir += 1

    # 테스트 이미지 디렉토리에 이미지가 충분하지 않으면 에러 메시지를 보여주고 false 리턴
    if numImagesInTestDir < MIN_NUM_IMAGES_REQUIRED_FOR_TESTING:
        print("에러: 테스트 하기에 최소 필요로 하는 이미지 수인 " + str(MIN_NUM_IMAGES_REQUIRED_FOR_TESTING) + " 개가 필요합니다. - " + TEST_IMAGES_DIR)
        print("테스트 이미지를 충분하게 넣어 주세요.")
        return False

    return True

#######################################################################################################################
def prepare_file_system():
    # TensorBoard에 대해 요약한 자료를 넣을 디렉토리 설정
    # tf.gfile.Exists() : 경로의 존재 여부 확인
    if tf.gfile.Exists(TENSORBOARD_DIR):
        # tf.gfile.DeleteRecursively(): 들어가 있는 로그 파일 삭제
        tf.gfile.DeleteRecursively(TENSORBOARD_DIR)

    # tf.gfile.MakeDirs(): 디렉터리 생성.
    tf.gfile.MakeDirs(TENSORBOARD_DIR)
    # 디렉토리 하위 폴더(train, validation)가 존재 하지 않을 경우 폴더 생성
    if INTERMEDIATE_STORE_FREQUENCY > 0:
        makeDirIfDoesNotExist(INTERMEDIATE_OUTPUT_GRAPHS_DIR)
    return

#######################################################################################################################
def makeDirIfDoesNotExist(dir_name):
    # 디렉토리에 하위 폴더가 존재하는지 확인
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

#######################################################################################################################
def create_model_info(architecture):
    """
    모델 아키텍처의 이름을 고려하여, 그것에 대한 정보를 반환한다.

    전송 학습을 이용해 재교육할 수 있는 베이스 이미지 인식의 사전 학습된 모델이 서로 다르며,
    이 기능은 모델 이름에서 모델과 함께 다운로드하여 학습하는 데 필요한 속성으로 변환된다.

    Args:
        아키텍쳐: 사용하려는 모델 아키텍쳐 이름

    Returns:
        모델에 대한 정보 또는 이름이 인식되지 않음.

    Raises:
        ValueError: 아키텍쳐 이름이 옳지 않은 경우
    """
    # architecture.lower() : architecture를 소문자로 변환하여 반환
    architecture = architecture.lower()
    is_quantized = False
    # 'inception_v3'일 경우
    if architecture == 'inception_v3':
        # pylint : Python 코드의 오류를 검사하고 적절한 Python 코딩 패턴을 권장하며 널리 사용되는 도구 (enable=line-too-long)
        data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        bottleneck_tensor_name = 'pool_3/_reshape:0'
        bottleneck_tensor_size = 2048
        input_width = 299
        input_height = 299
        input_depth = 3
        resized_input_tensor_name = 'Mul:0'
        model_file_name = 'classify_image_graph_def.pb'
        input_mean = 128
        input_std = 128

    return {'data_url': data_url, 'bottleneck_tensor_name': bottleneck_tensor_name, 'bottleneck_tensor_size': bottleneck_tensor_size,
            'input_width': input_width, 'input_height': input_height, 'input_depth': input_depth, 'resized_input_tensor_name': resized_input_tensor_name,
            'model_file_name': model_file_name, 'input_mean': input_mean, 'input_std': input_std, 'quantize_layer': is_quantized, }

#######################################################################################################################
def downloadModelIfNotAlreadyPresent(data_url):
    """
    tar 파일에서 모델을 다운로드 및 추출.

    만약 사용하려는 모델이 존재하지 않으면, TensorFlow.org 웹사이트에서 모델을 다운로드하여 디렉토리에 압축 해제.

    Args:
        data_url: 모델이 들어 있는 tar 파일의 웹 위치.
    """
    # 아키텍쳐 다운받을 디렉토리 장소
    dest_directory = MODEL_DIR
    # 디렉토리 존재 하지 않을 경우
    if not os.path.exists(dest_directory):
        # 디렉토리 생성
        os.makedirs(dest_directory)

    # 다운받는 주소 마지막 부분 - 모델이름
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)

    if not os.path.exists(filepath):
        # 파일 다운 현황 출력
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> 다운로드중 %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()


        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        # os.stat() = 파일 정보 가져옴
        statinfo = os.stat(filepath)
        tf.logging.info('다운로드 성공 ' + str(filename) + ', 파일 크기 = ' + str(statinfo.st_size) + ' bytes')
        print('파일 추출 : ', filepath)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    else:
        print('모델이 대상 디렉토리에 존재하지 않아 추출되지 않았거나 다운로드 되지 않았습니다.')

#######################################################################################################################
def create_model_graph(model_info):
    """"
    저장된 그래프 디렉토리에서 그래프를 생성하고 그래프 객체를 리턴.

    Args:
        model_info: 모델 아키텍처에 대한 정보

    Returns:
        학습된 인셉션 네트워크를 담고 있는 그래프와 여러 텐서.
    """
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(MODEL_DIR, model_info['model_file_name'])
        print('Model path: ', model_path)
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(graph_def, name='', return_elements=[model_info['bottleneck_tensor_name'], model_info['resized_input_tensor_name'],]))
        # end with
    # end with
    return graph, bottleneck_tensor, resized_input_tensor
# end function

#######################################################################################################################
def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """
    Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
        image_dir: String path to a folder containing subfolders of images.
        testing_percentage: Integer percentage of the images to reserve for tests.
        validation_percentage: Integer percentage of images reserved for validation.

    Returns:
        A dictionary containing an entry for each label subfolder, with images split
        into training, testing, and validation sets within each label.
    """

    # if the image directory does not exist, log an error and bail
    if not gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    # end if

    # create an empty dictionary to store the results
    result = {}

    # get a list of the sub-directories of the image directory
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

    # for each directory in the sub-directories list . . .
    is_root_dir = True
    for sub_dir in sub_dirs:
        # if we're on the 1st (root) directory, mark our boolean for that as false for the next time around and go back to the top of the for loop
        if is_root_dir:
            is_root_dir = False
            continue
        # end if

        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        # end if

        # ToDo: This section should be refactored.  The right way to do this would be to get a list of the files that are
        # ToDo: there then append (extend) those, not to get the name except the extension, then append an extension,
        # ToDo: this (current) way is error prone of the original file has an upper case or mixed case extension

        extensions = ['jpg', 'jpeg']
        file_list = []
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        # end for

        # if the file list is empty at this point, log a warning and bail
        if not file_list:
            tf.logging.warning('No files found')
            continue
        # end if

        # if the length of the file list is less than 20 or more than the max number, log an applicable warning (do not return, however)
        if len(file_list) < 20:
            tf.logging.warning('WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning('WARNING: Folder {} has more than {} images. Some images will never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        # end if

        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # 이미지를 넣을 수 있도록 디코딩할 때 파일 이름에 '_noash_' 뒤에 있는 다른 모든 것을 원하며, 데이터 세트 작성자는 서로 가까운 변형인 사진을 그룹화하는 방법을 가지고 있다. 
            # 예를 들어, 이것은 식물 질병 데이터 세트에서 동일한 잎의 여러 사진을 그룹화하기 위해 사용된다.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # 이것은 좀 신기해 보이지만, 우리는 이 파일이 훈련, 시험 또는 유효성 검사 세트로 들어가야 하는지 결정해야 하며, 우리는 나중에 더 많은 파일이 추가되더라도 기존 파일을 동일한 세트로 유지하기를 원한다.  
            # 그러기 위해서는 파일 이름 그 자체만으로 안정적인 결정 방법이 필요하기 때문에, 그것을 해시하고 나서 그것을 이용하여 그것을 할당하는 데 사용하는 확률 평가표를 생성한다.
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_IMAGES_PER_CLASS + 1)) * (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
            # end if
        result[label_name] = {'dir': dir_name, 'training': training_images, 'testing': testing_images, 'validation': validation_images,}
    return result
# end function

#######################################################################################################################
def add_jpeg_decoding(input_width, input_height, input_depth, input_mean, input_std):
    """
    Adds operations that perform JPEG decoding and resizing to the graph..
    Args:
        input_width: Desired width of the image fed into the recognizer graph.
        input_height: Desired width of the image fed into the recognizer graph.
        input_depth: Desired channels of the image fed into the recognizer graph.
        input_mean: Pixel value that should be zero in the image for the graph.
        input_std: How much to divide the pixel values by before recognition.

    Returns:
        Tensors for the node to feed JPEG data into, and the output of the preprocessing steps.
    """
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return jpeg_data, mul_image
# end function

#######################################################################################################################
def add_input_distortions(flip_left_right, random_crop, random_scale, random_brightness, input_width, input_height,
                          input_depth, input_mean, input_std):
    """
    Creates the operations to apply the specified distortions.

    During training it can help to improve the results if we run the images
    through simple distortions like crops, scales, and flips. These reflect the
    kind of variations we expect in the real world, and so can help train the
    model to cope with natural data more effectively. Here we take the supplied
    parameters and construct a network of operations to apply them to an image.

    Cropping
    ~~~~~~~~

    Cropping is done by placing a bounding box at a random position in the full
    image. The cropping parameter controls the size of that box relative to the
    input image. If it's zero, then the box is the same size as the input and no
    cropping is performed. If the value is 50%, then the crop box will be half the
    width and height of the input. In a diagram it looks like this:

    <       width         >
    +---------------------+
    |                     |
    |   width - crop%     |
    |    <      >         |
    |    +------+         |
    |    |      |         |
    |    |      |         |
    |    |      |         |
    |    +------+         |
    |                     |
    |                     |
    +---------------------+

    Scaling
    ~~~~~~~

    Scaling is a lot like cropping, except that the bounding box is always
    centered and its size varies randomly within the given range. For example if
    the scale percentage is zero, then the bounding box is the same size as the
    input and no scaling is applied. If it's 50%, then the bounding box will be in
    a random range between half the width and height and full size.

    Args:
        flip_left_right: Boolean whether to randomly mirror images horizontally.
        random_crop: Integer percentage setting the total margin used around the
        crop box.
        random_scale: Integer percentage of how much to vary the scale by.
        random_brightness: Integer range to randomly multiply the pixel values by.
        graph.
        input_width: Horizontal size of expected input image to model.
        input_height: Vertical size of expected input image to model.
        input_depth: How many channels the expected input image should have.
        input_mean: Pixel value that should be zero in the image for the graph.
        input_std: How much to divide the pixel values by before recognition.

    Returns:
        The jpeg input layer and the distorted result tensor.
    """

    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(tensor_shape.scalar(), minval=1.0, maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, input_width)
    precrop_height = tf.multiply(scale_value, input_height)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d, precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    cropped_image = tf.random_crop(precropped_image_3d, [input_height, input_width, input_depth])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    # end if
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(), minval=brightness_min, maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    offset_image = tf.subtract(brightened_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')
    return jpeg_data, distort_result
# end function

#######################################################################################################################
def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, architecture):
    """
    Ensures all the training, testing, and validation bottlenecks are cached.

    Because we're likely to read the same image multiple times (if there are no distortions applied during training) it
    can speed things up a lot if we calculate the bottleneck layer values once for each image during preprocessing,
    and then just read those cached values repeatedly during training. Here we go through all the images we've found,
    calculate those values, and save them off.

    Args:
        sess: The current active TensorFlow Session.
        image_lists: Dictionary of training images for each label.
        image_dir: Root folder string of the subfolders containing the training images.
        bottleneck_dir: Folder string holding cached files of bottleneck values.
        jpeg_data_tensor: Input tensor for jpeg data from file.
        decoded_image_tensor: The output of decoding and resizing the image.
        resized_input_tensor: The input node of the recognition graph.
        bottleneck_tensor: The penultimate output layer of the graph.
        architecture: The name of the model architecture.

    Returns:
        Nothing.
    """
    how_many_bottlenecks = 0
    makeDirIfDoesNotExist(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category, bottleneck_dir,
                                         jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, architecture)
            # end for
            how_many_bottlenecks += 1
            if how_many_bottlenecks % 100 == 0:
                tf.logging.info(str(how_many_bottlenecks) + ' bottleneck files created.')
            # end if
# end function

#######################################################################################################################
def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor, bottleneck_tensor, architecture):
    """
    Retrieves or calculates bottleneck values for an image.

    If a cached version of the bottleneck data exists on-disk, return that, otherwise calculate the data and save it to disk for future use.

    Args:
        sess: The current active TensorFlow Session.
        image_lists: Dictionary of training images for each label.
        label_name: Label string we want to get an image for.
        index: Integer offset of the image we want. This will be modulo-ed by the available number of images for the label, so it can be arbitrarily large.
        image_dir: Root folder string of the subfolders containing the training images.
        category: Name string of which set to pull images from - training, testing, or validation.
        bottleneck_dir: Folder string holding cached files of bottleneck values.
        jpeg_data_tensor: The tensor to feed loaded jpeg data into.
        decoded_image_tensor: The output of decoding and resizing the image.
        resized_input_tensor: The input node of the recognition graph.
        bottleneck_tensor: The output tensor for the bottleneck values.
        architecture: The name of the model architecture.

    Returns:
        Numpy array of values produced by the bottleneck layer for the image.
    """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    makeDirIfDoesNotExist(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category, architecture)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor, bottleneck_tensor)
    # end if

    # read in the contents of the bottleneck file as one big string
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneckBigString = bottleneck_file.read()
    # end with

    bottleneckValues = []
    errorOccurred = False
    try:
        # split the bottleneck file contents read in as one big string into individual float values
        bottleneckValues = [float(individualString) for individualString in bottleneckBigString.split(',')]
    except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')
        errorOccurred = True
    # end try

    if errorOccurred:
        # if an error occurred above, create (or re-create) the bottleneck file
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess,
                               jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor)

        # read in the contents of the newly created bottleneck file
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneckBigString = bottleneck_file.read()
        # end with

        # split the bottleneck file contents read in as one big string into individual float values again
        bottleneckValues = [float(individualString) for individualString in bottleneckBigString.split(',')]
    # end if
    return bottleneckValues
# end function

#######################################################################################################################
def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category, architecture):
    """"
    Returns a path to a bottleneck file for a label at the given index.

    Args:
        image_lists: Dictionary of training images for each label.
        label_name: Label string we want to get an image for.
        index: Integer offset of the image we want. This will be moduloed by the
        available number of images for the label, so it can be arbitrarily large.
        bottleneck_dir: Folder string holding cached files of bottleneck values.
        category: Name string of set to pull images from - training, testing, or
        validation.
        architecture: The name of the model architecture.

    Returns:
        File system path string to an image that meets the requested parameters.
    """
    return get_image_path(image_lists, label_name, index, bottleneck_dir, category) + '_' + architecture + '.txt'
# end function

#######################################################################################################################
def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
    """Create a single bottleneck file."""
    tf.logging.info('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index, image_dir, category)
    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    # end if
    image_data = gfile.FastGFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path, str(e)))
    # end try

    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)
    # end with
# end function

#######################################################################################################################
def run_bottleneck_on_image(sess, image_data, image_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor):
    """
    Runs inference on an image to extract the 'bottleneck' summary layer.
    Args:
        sess: Current active TensorFlow Session.
        image_data: String of raw JPEG data.
        image_data_tensor: Input data layer in the graph.
        decoded_image_tensor: Output of initial image resizing and preprocessing.
        resized_input_tensor: The input node of the recognition graph.
        bottleneck_tensor: Layer before the final softmax.

    Returns:
        Numpy array of bottleneck values.
    """
    # First decode the JPEG image, resize it, and rescale the pixel values.
    resized_input_values = sess.run(decoded_image_tensor, {image_data_tensor: image_data})
    # Then run it through the recognition network.
    bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values
# end function

#######################################################################################################################
def get_image_path(image_lists, label_name, index, image_dir, category):
    """"
    Returns a path to an image for a label at the given index.

    Args:
        image_lists: Dictionary of training images for each label.
        label_name: Label string we want to get an image for.
        index: Int offset of the image we want. This will be moduloed by the available number of images for the label, so it can be arbitrarily large.
        image_dir: Root folder string of the subfolders containing the training images.
        category: Name string of set to pull images from - training, testing, or validation.

    Returns:
        File system path string to an image that meets the requested parameters.
    """
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    # end if
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    # end if
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.', label_name, category)
    # end if
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path
# end function

#######################################################################################################################
def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor, bottleneck_tensor_size, quantize_layer):
    """
    Adds a new softmax and fully-connected layer for training.

    We need to retrain the top layer to identify our new classes, so this function
    adds the right operations to the graph, along with some variables to hold the
    weights, and then sets up all the gradients for the backward pass.

    The set up for the softmax and fully-connected layers is based on:
    https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

    Args:
        class_count: Integer of how many categories of things we're trying to recognize.
        final_tensor_name: Name string for the new final node that produces results.
        bottleneck_tensor: The output of the main CNN graph.
        bottleneck_tensor_size: How many entries in the bottleneck vector.
        quantize_layer: Boolean, specifying whether the newly added layer should be quantized.

    Returns:
        The tensors for the training and cross entropy results, and tensors for the bottleneck input and ground truth input.
    """
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(bottleneck_tensor, shape=[None, bottleneck_tensor_size], name='BottleneckInputPlaceholder')
        ground_truth_input = tf.placeholder(tf.int64, [None], name='GroundTruthInput')
    # end with

    # Organizing the following ops as `final_training_ops` so they're easier to see in TensorBoard
    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        quantized_layer_weights = None
        quantized_layer_biases = None
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal([bottleneck_tensor_size, class_count], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')
            if quantize_layer:
                quantized_layer_weights = quant_ops.MovingAvgQuantize(layer_weights, is_training=True)
                attachTensorBoardSummaries(quantized_layer_weights)
            # end if

            # this comment is necessary to suppress an unnecessary PyCharm warning
            # noinspection PyTypeChecker
            attachTensorBoardSummaries(layer_weights)
        # end with
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            if quantize_layer:
                quantized_layer_biases = quant_ops.MovingAvgQuantize(layer_biases, is_training=True)
                attachTensorBoardSummaries(quantized_layer_biases)
            # end if

            # this comment is necessary to suppress an unnecessary PyCharm warning
            # noinspection PyTypeChecker
            attachTensorBoardSummaries(layer_biases)
        # end with
        with tf.name_scope('Wx_plus_b'):
            if quantize_layer:
                logits = tf.matmul(bottleneck_input, quantized_layer_weights) + quantized_layer_biases
                logits = quant_ops.MovingAvgQuantize(logits, init_min=-32.0, init_max=32.0, is_training=True, num_bits=8,
                                                     narrow_range=False, ema_decay=0.5)
                tf.summary.histogram('pre_activations', logits)
            else:
                logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
                tf.summary.histogram('pre_activations', logits)
            # end if
        # end with
    # end with
    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels=ground_truth_input, logits=logits)
    # end with

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        train_step = optimizer.minimize(cross_entropy_mean)
    # end with

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)
# end function

#######################################################################################################################
def attachTensorBoardSummaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # end with
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
    # end with
# end function

#######################################################################################################################
def add_evaluation_step(result_tensor, ground_truth_tensor):
    """
    Inserts the operations we need to evaluate the accuracy of our results.
    Args:
        result_tensor: The new final node that produces results.
        ground_truth_tensor: The node we feed ground truth data into.
    Returns:
        Tuple of (evaluation step, prediction).
    """

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, ground_truth_tensor)
        # end with
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # end with
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction
# end function

#######################################################################################################################
def get_random_distorted_bottlenecks(sess, image_lists, how_many, category, image_dir, input_jpeg_tensor, distorted_image,
                                     resized_input_tensor, bottleneck_tensor):
    """
    Retrieves bottleneck values for training images, after distortions.

    If we're training with distortions like crops, scales, or flips, we have to recalculate the full model for every image,
    and so we can't use cached bottleneck values. Instead we find random images for the requested category, run them through
    the distortion graph, and then the full graph to get the bottleneck results for each.

    Args:
        sess: Current TensorFlow Session.
        image_lists: Dictionary of training images for each label.
        how_many: The integer number of bottleneck values to return.
        category: Name string of which set of images to fetch - training, testing, or validation.
        image_dir: Root folder string of the subfolders containing the training images.
        input_jpeg_tensor: The input layer we feed the image data to.
        distorted_image: The output node of the distortion graph.
        resized_input_tensor: The input node of the recognition graph.
        bottleneck_tensor: The bottleneck output layer of the CNN graph.

    Returns:
        List of bottleneck arrays and their corresponding ground truths.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir, category)
        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        # end if
        jpeg_data = gfile.FastGFile(image_path, 'rb').read()
        # Note that we materialize the distorted_image_data as a numpy array before
        # sending running inference on the image. This involves 2 memory copies and
        # might be optimized in other implementations.
        distorted_image_data = sess.run(distorted_image, {input_jpeg_tensor: jpeg_data})
        bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: distorted_image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        bottlenecks.append(bottleneck_values)
        ground_truths.append(label_index)
    # end for
    return bottlenecks, ground_truths
# end function

#######################################################################################################################
def get_random_cached_bottlenecks(sess, image_lists, how_many, category, bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor, bottleneck_tensor, architecture):
    """
    Retrieves bottleneck values for cached images.

    If no distortions are being applied, this function can retrieve the cached bottleneck values directly from disk for
    images. It picks a random set of images from the specified category.

    Args:
        sess: Current TensorFlow Session.
        image_lists: Dictionary of training images for each label.
        how_many: If positive, a random sample of this size will be chosen.  If negative, all bottlenecks will be retrieved.
        category: Name string of which set to pull from - training, testing, or validation.
        bottleneck_dir: Folder string holding cached files of bottleneck values.
        image_dir: Root folder string of the subfolders containing the training images.
        jpeg_data_tensor: The layer to feed jpeg image data into.
        decoded_image_tensor: The output of decoding and resizing the image.
        resized_input_tensor: The input node of the recognition graph.
        bottleneck_tensor: The bottleneck output layer of the CNN graph.
        architecture: The name of the model architecture.

    Returns:
        List of bottleneck arrays, their corresponding ground truths, and the relevant filenames.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        # Retrieve a random sample of bottlenecks.
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, image_dir, category, bottleneck_dir,
                                                  jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, architecture)
            bottlenecks.append(bottleneck)
            ground_truths.append(label_index)
            filenames.append(image_name)
        # end for
    else:
        # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)
                bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, image_dir, category, bottleneck_dir,
                                                      jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, architecture)
                bottlenecks.append(bottleneck)
                ground_truths.append(label_index)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames
# end function

#######################################################################################################################
def save_graph_to_file(sess, graph, graph_file_name):
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [FINAL_TENSOR_NAME])
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    # end with
    return
# end function

#######################################################################################################################
if __name__ == '__main__':
    main()
