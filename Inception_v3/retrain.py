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
MAX_NUM_IMAGES_PER_CLASS = 2 ** 57 - 1

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
# 좋은 학습 데이터를 수집했는지를 검증하기 위해 이 작업부터 시작하는 것이 좋으나, 연구 제한 플랫폼에 배치하려면 MobildNet 모델로 "--architecture" 플래그를 사용할 수 있다.

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
    # 저장된 그래프 디렉토리에서 그래프를 생성하고 그래프 객체를 리턴.
    print("모델 그래프 생성...")
    graph, bottleneck_tensor, resized_image_tensor = (create_model_graph(model_info))

    # 폴더 구조를 보고 모든 이미지의 목록을 생성.
    print("이미지 리스트 생성...")
    image_lists = create_image_lists(TRAINING_IMAGES_DIR, TESTING_PERCENTAGE, VALIDATION_PERCENTAGE)
    # .keys(): image_lists key만을 모아서 list 객체를 리턴.
    class_count = len(image_lists.keys())
    if class_count == 0:
        tf.logging.error( TRAINING_IMAGES_DIR + ' 위치에서 올바른 이미지 파일을 찾지 못하였습니다.')
        return -1

    if class_count == 1:
        tf.logging.error(TRAINING_IMAGES_DIR + ' 위치에서 올바른 이미지 폴더를 하나만 찾았습니다 - 분류하기 위해서는 여러 이미지 폴더가 필요합니다.')
        return -1

    # 왜곡 명령 라인 플래그가 설정되었는지 확인
    doDistortImages = False
    if (FLIP_LEFT_RIGHT == True or RANDOM_CROP != 0 or RANDOM_SCALE != 0 or RANDOM_BRIGHTNESS != 0):
        doDistortImages = True

    print("세션 시작...")
    with tf.Session(graph=graph) as sess:
        # 이미지 디코딩 서브 그래프 설정.
        print("jpeg/jpg 디코딩 시작...")
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding( model_info['input_width'],
                                                                    model_info['input_height'],
                                                                    model_info['input_depth'],
                                                                    model_info['input_mean'],
                                                                    model_info['input_std'])
        print("디코딩 데이터 캐싱...")
        distorted_jpeg_data_tensor = None
        distorted_image_tensor = None
        # 이미지 왜곡을 했는지 확인하는 작업  
        # -> 왜곡을 함으로써 같은 이미지 파일에 여러 왜곡이 일어나는 상황(플립, 크롭, 스케일 등)을 만듦으로써 정확성을 높이기 위하여 사용
        if doDistortImages:
            # 이미지 왜곡을 발생
            (distorted_jpeg_data_tensor, distorted_image_tensor) = add_input_distortions(FLIP_LEFT_RIGHT, RANDOM_CROP, RANDOM_SCALE,
                                                                                        RANDOM_BRIGHTNESS, model_info['input_width'],
                                                                                        model_info['input_height'], model_info['input_depth'],
                                                                                        model_info['input_mean'], model_info['input_std'])
        else:
            # 'bottlenect' 이미지 확인 후 캐시가 올바르게 되어있는지 확인
            cache_bottlenecks(sess, image_lists, TRAINING_IMAGES_DIR, BOTTLENECK_DIR, jpeg_data_tensor, decoded_image_tensor,
                            resized_image_tensor, bottleneck_tensor, ARCHITECTURE)

        # 학습할 새로운 레이어 추가
        print("최종 학습 레이어 추가...")
        (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = add_final_training_ops(len(image_lists.keys()),
                                                                                                                FINAL_TENSOR_NAME,
                                                                                                                bottleneck_tensor,
                                                                                                                model_info['bottleneck_tensor_size'],
                                                                                                                model_info['quantize_layer'])

        # 레이어 정확성을 평가
        print("최종 학습 레이어에 대한 평가 추가...")
        evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)

        # 모든 데이터 합쳐서 tensorboard_dir 저장
        print("TensorBoard 정보 작성...")
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(TENSORBOARD_DIR + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(TENSORBOARD_DIR + '/validation')

        # 가중치 초기값으로 설정
        init = tf.global_variables_initializer()
        sess.run(init)

        # 학습 시작
        print("학습 시작...")
        for i in range(HOW_MANY_TRAINING_STEPS):
            # bottleneck 값 가져오기(할 때 마다 갱신)
            # 왜곡이 적용된 시간 또는 디스크에 저장된 캐시 시간.
            if doDistortImages:
                (train_bottlenecks, train_ground_truth) = get_random_distorted_bottlenecks(sess, image_lists, TRAIN_BATCH_SIZE, 'training',
                                                                                        TRAINING_IMAGES_DIR, distorted_jpeg_data_tensor,
                                                                                        distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
            else:
                (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(sess, image_lists, TRAIN_BATCH_SIZE, 'training',
                                                                                        BOTTLENECK_DIR, TRAINING_IMAGES_DIR, jpeg_data_tensor,
                                                                                        decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                                                                                        ARCHITECTURE)

            # bottleneck과 ground truth를 그래프에 입력하고 학습 실행하십시오.
            # Merged로 TensorBoard의 학습 캡처.
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

            # 중간 결과 저장
            intermediate_frequency = INTERMEDIATE_STORE_FREQUENCY

            if (intermediate_frequency > 0 and (i % intermediate_frequency == 0) and i > 0):
                intermediate_file_name = (INTERMEDIATE_OUTPUT_GRAPHS_DIR + 'intermediate_' + str(i) + '.pb')
                tf.logging.info('중간 결과 저장 : ' + intermediate_file_name)
                save_graph_to_file(sess, graph, intermediate_file_name)

        # 학습을 마치고 테스트 데이터로 최종 테스트 평가를 실행
        print("테스트 시작...")
        test_bottlenecks, test_ground_truth, test_filenames = (get_random_cached_bottlenecks(sess, image_lists, TEST_BATCH_SIZE, 'testing', BOTTLENECK_DIR,
                                                                                            TRAINING_IMAGES_DIR, jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                                                                                            bottleneck_tensor, ARCHITECTURE))
        test_accuracy, predictions = sess.run([evaluation_step, prediction], feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        tf.logging.info('테스트 결과 = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))

        if PRINT_MISCLASSIFIED_TEST_IMAGES:
            tf.logging.info('=== 분류되지 않은 테스트 이미지 ===')
            for i, test_filename in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i]:
                    tf.logging.info('%70s  %s' % (test_filename, list(image_lists.keys())[predictions[i]]))

        # 학습된 그래프와 라벨을 저장된 가중치로 저장
        print("웨이트로 학습된 그래프 및 레이블 저장")
        save_graph_to_file(sess, graph, OUTPUT_GRAPH)
        with gfile.FastGFile(OUTPUT_LABELS, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')

        print("학습 종료...")

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

    # 각 학습 서브 디렉토리에 학습 이미지들을 채우기.
    for trainingSubDir in trainingSubDirs:
        # 현재 서브 디렉토리에 이미지가 몇개 있는지 확인하기
        for fileName in os.listdir(trainingSubDir.loc):
            if fileName.endswith(".jpeg") or fileName.endswith(".jpg"):
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

    전송 학습을 이용해 재학습할 수 있는 베이스 이미지 인식의 사전 학습된 모델이 서로 다르며,
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
    # tf.Graph(): 데이터 흐름 그래프로 표시되는 TensorFlow 연산.
    # as_default(): 그래프를 기본 그래프로 만드는 context 관리자를 리턴.
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(MODEL_DIR, model_info['model_file_name'])
        print('Model path: ', model_path)
        # gfile.FastGFile(): 쓰레드 잠금이 없는 파일 I/O
        with gfile.FastGFile(model_path, 'rb') as f:
            # tf.GraphDef(): 프로토콜 이미지 
            graph_def = tf.GraphDef()
            # ParseFromString(): 주어진 문자열에서 메시지를 구문 분석 
            graph_def.ParseFromString(f.read())
            # tf.import_graph_def(): graph_def에서 현재 기본 Graph로 그래프 가져오기(비평가된 인수)
            bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(graph_def, name='', return_elements=[model_info['bottleneck_tensor_name'], model_info['resized_input_tensor_name'],]))

    return graph, bottleneck_tensor, resized_input_tensor

#######################################################################################################################
def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """
    파일 시스템에서 학습 이미지 목록 생성

    이미지 디렉토리의 하위 폴더를 분석하고 이를 training, test 및 validation 셋으로 분할한 다음 
    각 라벨과 해당 경로에 대한 이미지 리스트를 설명하는 데이터 구조를 리턴.

    Args:
        image_dir: 이미지의 하위 폴더를 포함하는 폴더의 문자열 경로.
        testing_percentage: 테스트를 위해 사용될 이미지 비율.
        validation_percentage: validaion을 위해 사용될 이미지 비율.

    Returns:
        하위 폴더의 항목에 포함된 딕셔너리 중 training, test 및 validation set으로 이미지를 분할.
    """

    # 만일 이미지 파일 디렉토리가 존재하지 않으면 에러 처리 후 리턴.
    if not gfile.Exists(image_dir):
        tf.logging.error("이미지 파일 '" + image_dir + "'를 찾지 못하였습니다.")
        return None

    # 결과를 저장하기 위해 빈 폴더 생성.
    result = {}

    # 이미지 폴더 하위 폴더 목록 가져오기
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

    # 각 이미지 폴더 안 하위 폴더 리스트 확인
    is_root_dir = True
    for sub_dir in sub_dirs:
        # 하위 폴더를 체크했는지 확인하기 위한 부분. (폴더를 체크하면 False로 변경)
        if is_root_dir:
            is_root_dir = False
            continue
        # os.path.basename(): 경로의 최종 구성 요소 return
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue

        extensions = ['jpg', 'jpeg']
        file_list = []
        tf.logging.info("'" + dir_name + "에 있는 이미지 찾는 중...'")
        for extension in extensions:
            # os.path.join(): 두 개 이상의 이름 구성 요소를 결합한 후 디폴트 값으로 '/'가 이름 중간에 삽입된다.
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            # .extend(A): A 요소 추가
            # gfile.Glob(): 지정된 패턴과 일치하는 파일 return 
            file_list.extend(gfile.Glob(file_glob))

        # 이 시점에서 파일 목록이 비어 있는 경우 경고 로그 및 에러 메시지 표시
        if not file_list:
            tf.logging.warning('파일을 찾을 수 없습니다')
            continue

        # 파일 리스트 길이가 최대값보다 20보다 작거나 더 크면 해당 경고 로그(return X).
        if len(file_list) < 20:
            tf.logging.warning('경고: 이미지가 20장 미만입니다. 이미지를 더 채워 넣어주세요.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning('경고: 폴더 {} 에 있는 이미지가 {} 보다 큽니다. 몇몇 이미지가 해당되지 않을 수 있습니다.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))

        # re.sub(pattern, repl, string): string에서 pattern과 매치하는 텍스트를 repl로 치환한다
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []

        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # 이미지 파일 속 여러 중복되는 데이터가 있을 때 그룹화하기 위해 쓰여지는 코드
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # 파일이 train, test, validation set로 들어가야 하는지 결정해야 하며, 나중에 더 많은 파일이 추가되더라도 기존 파일을 동일한 세트로 유지하기 위해.
            # 이를 유지하기 위해서 파일 이름을 안정적으로 결정해야 함으로 해시하고 나서 이를 이용하여 이름을 할당하는 데 사용하는 확률 평가표를 생성한다.
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_IMAGES_PER_CLASS + 1)) * (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {'dir': dir_name, 'training': training_images, 'testing': testing_images, 'validation': validation_images,}

    return result

#######################################################################################################################
def add_jpeg_decoding(input_width, input_height, input_depth, input_mean, input_std):
    """
    Adds operations that perform JPEG decoding and resizing to the graph..
    디코딩 및 리사이즈 된 JPEG 파일을 그래프에 추가
    Args:
        input_width: 그래프에 입력된 이미지가 원하는 너비. (현재값 299)
        input_height: 그래프에 입력된 이미지가 원하는 높이. (현재값 299)
        input_depth: 그래프에 입력된 이미지가 원하는 채널. (현재값 3)
        input_mean: 그래프 이미지에서 0이어야 하는 픽셀 값. (현재값 128)
        input_std: recognition 전 픽셀 값을 나눈 값. (현재값 128)

    Returns:
        JPEG 데이터를 제공하는 텐서와 사전 처리 단계 출력.
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

#######################################################################################################################
def add_input_distortions(flip_left_right, random_crop, random_scale, random_brightness, 
                        input_width, input_height, input_depth, input_mean, input_std):
    """
    특정 왜곡을 적용시키기 위한 작업.

    학습 하는 동안 크롭, 스케일, 플립과 같은 왜곡을 통하여 이미지를 실행하면 정확성을 높이는데 도움이 된다.
    사용자들이 실제로 사용할 때 높은 품질의 이미지만을 제공하지 않기에 이 왜곡은 학습 모델이 여러 데이터에 대해 효과적으로 대처하도록 만든다.

    크롭

    크롭은 전체 이미지 중 랜덤 위치에 bounding box를 배치하여 수행된다. 매개변수는 입력 이미지에 대한 box 크기를 제어한다.
    0이면 입력 사이즈와 같은 크기로 자르지 않는다. 값이 50%이면 crop box는 입력 폭과 높이의 절반이다.

    Scaling
    ~~~~~~~
    스케일링을 통해 다차원의 값들을 비교 분석하기 쉽게 만들어주며, 
    자료의 오버플로우(overflow)나 언더플로우(underflow)를 방지 하고,
    독립 변수의 공분산 행렬의 조건수(condition number)를 감소시켜 최적화 과정에서의 안정성 및 수렴 속도를 향상 시킨다.

    bounding box가 항상 중심에 있고 크기가 주어진 범위 내에서 무작위로 다르다는 것을 제외하고, scaling은 cropping과 매우 비슷하다.
    예를 들어 scale 백분율이 0인 경우 bounding box는 입력과 크기가 같고 scale는 적용되지 않는다.
    50%라면 scaling box는 폭과 높이 그리고 전체 크기 사이의 임의의 범위에 있게 될 것이다.


    Args:
        flip_left_right: 수평 이미지를 랜덤으로 미러링할지 여부.
        random_crop: crop box 주위에 사용된 총 마진을 설정하는 정수 비율.
        random_scale: scale를 변경할 수 있는 정수 비율.
        random_brightness: 픽셀 값을 그래프로 랜덤하게 곱하는 정수 범위.
        input_width: 모델 이미지의 수평 크기.
        input_height: 모델 이미지의 수직 크기.
        input_depth: 입력 이미지 채널 수.
        input_mean: 그래프의 이미지에서 0이어야 하는 픽셀 값.
        input_std: 인식 전에 픽셀 값을 나눈 값.

    Returns:
        jpeg input layer와 왜곡된 결과 tesnor.
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
        
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(), minval=brightness_min, maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    offset_image = tf.subtract(brightened_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    distort_result = tf.expand_dims(mul_image, 0, name='DistortResult')

    return jpeg_data, distort_result

#######################################################################################################################
def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                    resized_input_tensor, bottleneck_tensor, architecture):
    """
    모든 training, test, validation 데이터가 캐시로 저장되는지 확인

    이유로는 같은 이미지를 여러 번 읽을 가능성이 있기 때문에(학습 중에 왜곡이 적용되지 않는다면) 
    사전 처리 중에 각 이미지에 대해 bottleneck이 존재하는 계층 값을 확인한 다음,
    훈련 중에 캐시된 값을 반복해서 읽기만 하면 속도가 빨라진다.

    Args:
        sess: 현재 사용하고 있는 TensorFlow 세션.
        image_lists: 각 라벨에 대한 학습 이미지 리스트.
        image_dir: 학습 이미지를 포함하는 폴더.
        bottleneck_dir: bottleneck 캐시된 파일을 보관하는 폴더 문자열.
        jpeg_data_tensor: 파일에서 jpeg 데이터에 대한 텐서 입력.
        decoded_image_tensor: 디코딩 및 리사이즈 된 이미지 출력. 
        resized_input_tensor: 인식 그래프의 입력 노드.
        bottleneck_tensor: 그래프의 2차 출력 계층.
        architecture: 모델 아키텍쳐 이름
    """

    how_many_bottlenecks = 0
    makeDirIfDoesNotExist(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category, bottleneck_dir,
                                        jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, architecture)
            how_many_bottlenecks += 1
            if how_many_bottlenecks % 100 == 0:
                tf.logging.info(str(how_many_bottlenecks) + ' bottleneck 파일 생성완료.')

#######################################################################################################################
def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir, category, bottleneck_dir, jpeg_data_tensor,
                            decoded_image_tensor, resized_input_tensor, bottleneck_tensor, architecture):
    """
    이미지 bottleneck 값 검색 또는 계산

    캐시된 bottleneck 데이터가 디스크에 있는 경우 이를 리턴하고, 그렇지 않으면 데이터를 계산하여 나중에 사용할 수 있도록 디스크에 저장

    Args:
        sess: 현재 사용하고 있는 텐서플로우 세션
        image_lists: 각 라벨에 대한 학습 이미지 위치
        label_name: 이미지를 가져올 레이블
        index: 원하는 이미지의 정수 오프셋. 사용할 수 있는 이미지 수에 의해 모듈화되므로 수가 상황에 따라 다름
        image_dir: 학습 이미지를 포함하는 폴더의 루트 폴더 이름
        category: 이미지를 가져올 카테고리 선택 - training, testing, or validation
        bottleneck_dir: bottleneck 내 캐시된 파일을 보관하는 폴더 문자열
        jpeg_data_tensor: 로드된 jpeg 데이터를 공급할 텐서
        decoded_image_tensor: 이미지 디코딩 및 리사이즈 된 이미지 출력
        resized_input_tensor: 인식 그래프의 입력 노드
        bottleneck_tensor: bottleneck 출력 텐서
        architecture: 아키텍쳐 모델 이름

    Returns:
        이미지 bottleneck 레이어에 의해 생성된 numpy 배열.
    """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    makeDirIfDoesNotExist(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category, architecture)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor,
                            decoded_image_tensor, resized_input_tensor, bottleneck_tensor)

    # bottleneckt 파일의 내용을 하나의 큰 문자열로 만들기
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneckBigString = bottleneck_file.read()

    bottleneckValues = []
    errorOccurred = False
    try:
        # 하나의 큰 문자열로 읽은 bottleneck 파일 내용을 개별 값으로 분할
        bottleneckValues = [float(individualString) for individualString in bottleneckBigString.split(',')]
    except ValueError:
        tf.logging.warning('올바르지 않는 값을 발견했습니다 - bottleneck 다시 만듭니다')
        errorOccurred = True

    if errorOccurred:
        # 위에서 오류가 발생한 경우 bottleneck 파일 생성
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess,
                            jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor)

        # 새로 만든 bottleneck 파일의 내용을 읽기
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneckBigString = bottleneck_file.read()

        # 하나의 큰 문자열로 읽은 bottleneck 파일 내용을 개별 값으로 분할
        bottleneckValues = [float(individualString) for individualString in bottleneckBigString.split(',')]

    return bottleneckValues

#######################################################################################################################
def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category, architecture):
    """"
    지정된 인덱스 레이블에 대한 bottleneck 파일 경로 리턴

    Args:
        image_lists: 각각 라벨에 대한 학습 이미지 경로
        label_name: 이미지를 가져올 레이블 이름
        index: 원하는 이미지의 정수 오프셋. 사용할 수 있는 이미지 수에 의해 모듈화되므로 수가 상황에 따라 다름
        bottleneck_dir: bottleneck에 캐시된 파일을 가지고 있는 폴더
        category: 이미지를 가져올 카테고리 선택 - training, testing, or validation
        architecture: 사용할 아키텍쳐 모델 이름

    Returns:
        요청된 파라미터를 충족하는 이미지에 대한 파일 시스템 경로
    """
    return get_image_path(image_lists, label_name, index, bottleneck_dir, category) + '_' + architecture + '.txt'

#######################################################################################################################
def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                        image_dir, category, sess, jpeg_data_tensor,
                        decoded_image_tensor, resized_input_tensor,
                        bottleneck_tensor):
    """
        단일 bottleneck 파일 생성
    """
    tf.logging.info('bottleneck 파일을 생성합니다 - ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index, image_dir, category)
    if not gfile.Exists(image_path):
        tf.logging.fatal('파일이 %s 이 존재하지 않습니다', image_path)

    image_data = gfile.FastGFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('파일을 처리하던 중 에러가 발생했습니다 - %s (%s)' % (image_path, str(e)))

    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)

#######################################################################################################################
def run_bottleneck_on_image(sess, image_data, image_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor):
    """
    Runs inference on an image to extract the 'bottleneck' summary layer.
    함수를 실행하여 'bottleneck' 레이어 추출
    Args:
        sess: 현재 활성화된 tensorflow 세션
        image_data: JPEG 데이터 이름
        image_data_tensor: 그래프 데이터 레이어 입력
        decoded_image_tensor: 초기 이미지 리사이즈 & preprocessing 결과
        resized_input_tensor: 인식된 그래프 입력 노드
        bottleneck_tensor: 소프트맥스보다 앞서 레이어를 쌓음

    Returns:
        bottleneck 값 - numpy 
    """
    # 먼저 JPEG 이미지를 디코딩하고 리사이즈 하고 픽셀 값의 스케일을 재조정.
    resized_input_values = sess.run(decoded_image_tensor, {image_data_tensor: image_data})
    # 인식 네트워크를 통해 실행.
    bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

#######################################################################################################################
def get_image_path(image_lists, label_name, index, image_dir, category):
    """"
    지정된 인덱스의 레이블 이미지에 대한 위치 리턴

    Args:
        image_lists: 학습 이미지와 각각 라벨에 대한 위치
        label_name: 이미지를 가져올 레이블 문자열
        index: 원하는 이미지의 내부 오프셋. 라벨에 사용하는 이미지 수에 따라 바뀌므로 값이 일정치 않다.
        image_dir: 학습 이미지를 포함하는 폴더의 루트 폴더 문자열.
        category: 가져올 이미지를 포함하고 있는 문자열 이름 - training, testing, or validation.

    Returns:
        요청된 매개 변수를 충족하는 이미지에 대한 파일 시스템 경로
    """
    if label_name not in image_lists:
        tf.logging.fatal('라벨이 존재하지 않습니다 - %s.', label_name)

    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('카테고리가 존재하지 않습니다 - %s.', category)

    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('라벨 %s 카테고리 %s 에 존재하지 않습니다.', label_name, category)

    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


#######################################################################################################################
def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor, bottleneck_tensor_size, quantize_layer):
    """
    학습 할 소프트맥스와 fully-connected layer 추가.

    새로운 layer를 식별하기 위해 top layer를 재학습 해야한다.
    이를 위해 그래프에 연산을 추가하고, 가중치를 고정하기 위한 몇 가지 변수를 추가한다. 
    마지막으로 backward pass를 위한 gradient를 실시.

    Args:
        class_count: 인식할 이미지 리스트 갯수.
        final_tensor_name: 결과를 생성하는 최종 노드 이름.
        bottleneck_tensor: CNN 그래프.
        bottleneck_tensor_size: bottleneck 벡터 수.
        quantize_layer: 새로 추가된 레이어를 정량화해야 하는지 판단.

    Returns:
        학습 및 cross entropy 결과의 텐서. 
        bottleneck 입력과 ground truth 텐서.
    """
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(bottleneck_tensor, shape=[None, bottleneck_tensor_size], name='BottleneckInputPlaceholder')
        ground_truth_input = tf.placeholder(tf.int64, [None], name='GroundTruthInput')

    # TensorBoard에서 보다 쉽게 볼 수 있도록 "final_training_ops"로 구성
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

            # 불필요한 PyCharm 경고 & noinspection PyTypeChecker 억제
            attachTensorBoardSummaries(layer_weights)

        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            if quantize_layer:
                quantized_layer_biases = quant_ops.MovingAvgQuantize(layer_biases, is_training=True)
                attachTensorBoardSummaries(quantized_layer_biases)

            # 불필요한 PyCharm 경고 & noinspection PyTypeChecker 억제
            attachTensorBoardSummaries(layer_biases)

        with tf.name_scope('Wx_plus_b'):
            if quantize_layer:
                logits = tf.matmul(bottleneck_input, quantized_layer_weights) + quantized_layer_biases
                logits = quant_ops.MovingAvgQuantize(logits, init_min=-32.0, init_max=32.0, is_training=True, num_bits=8,
                                                    narrow_range=False, ema_decay=0.5)
                tf.summary.histogram('pre_activations', logits)
            else:
                logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
                tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels=ground_truth_input, logits=logits)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor)

#######################################################################################################################
def attachTensorBoardSummaries(var):
    """
    텐서 요약 첨부(텐서보드 시각화를 위해서)
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

#######################################################################################################################
def add_evaluation_step(result_tensor, ground_truth_tensor):
    """
    결과의 정확성 평가

    Args:
        result_tensor: 결과를 생성하는 최종 노드.
        ground_truth_tensor: ground truth 데이터 노드.

    Returns:
        평가, 예측 튜플
    """

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, ground_truth_tensor)

        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction

#######################################################################################################################
def get_random_distorted_bottlenecks(sess, image_lists, how_many, category, image_dir, input_jpeg_tensor, distorted_image,
                                    resized_input_tensor, bottleneck_tensor):
    """
    왜곡 작업 후 이미지 학습 bottleneck 값 확인

    만약 crop , scale, flip 같은 왜곡을 이용해 학습하면 모든 이미지에 대한 모델을 다시 계산해야 이는 캐시된 bottleneck 값을 사용할 수 없다.
    이를 해결하기 위해 요청된 범위에 대한 임의의 이미지를 찾아 왜곡 그래프 실행한 다음 전체 그래프에 각 범위에 대한 bottleneck 결과를 얻는다.

    Args:
        sess: 현재 텐서플로우 세션
        image_lists: 각 라벨에 대한 학습 이미지 리스트
        how_many: 사용할 bottleneck 값 개수
        category: 가져올 이미지 집합 이름 - training, testing, validation
        image_dir: 학습 이미지를 포함하는 하위 폴더의 루트 폴더 문자열
        input_jpeg_tensor: 이미지 데이터를 제공하는 입력 레이어
        distorted_image: 왜곡 그래프 출력 노드.
        resized_input_tensor: 인식된 그래프의 입력 노드.
        bottleneck_tensor: CNN 그래프 bottlenck 레이어.

    Returns:
        bottleneck 배열과 ground truth
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
            tf.logging.fatal('파일이 위치하지 않습니다 : %s', image_path)

        jpeg_data = gfile.FastGFile(image_path, 'rb').read()
        # 이미지에 대한 실행 추론을 전송하기 전에 distorted_image_data를 numpy 배열로 구체화.
        # 2개의 복사본을 생성하고 최적화.
        distorted_image_data = sess.run(distorted_image, {input_jpeg_tensor: jpeg_data})
        bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: distorted_image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        bottlenecks.append(bottleneck_values)
        ground_truths.append(label_index)

    return bottlenecks, ground_truths

#######################################################################################################################
def get_random_cached_bottlenecks(sess, image_lists, how_many, category, bottleneck_dir, image_dir, jpeg_data_tensor,
                                decoded_image_tensor, resized_input_tensor, bottleneck_tensor, architecture):
    """
    캐시된 이미지의 bottleneck 값 검색

    왜곡이 적용되지 않는 경우 이 기능은 디스크에서 이미지를 위해 캐시된 병목 현상 값을 직접 검색할 수 있다.
    지정된 범위에서 임의의 이미지를 선택한다.

    Args:
        sess: 현재 텐서플로우 세션
        image_lists: 각 라벨에 대한 학습 이미지 리스트
        how_many: true일 경우 랜덤 이미지 리스트가 선택.  false이면 bottleneck return
        category: 가져올 이름 문자열 - training, testing, validation
        bottleneck_dir: 캐시된 파일을 보관하는 폴더 문자열
        image_dir: 학습 이미지 폴더
        jpeg_data_tensor: jpeg 이미지 데이터 계층
        decoded_image_tensor: 이미지 디코딩 및 리사이즈 출력
        resized_input_tensor: 인식 그래프의 입력 노드
        bottleneck_tensor: CNN 그래프의 bottleneck 레이어
        architecture: 모델 아키텍쳐 이름

    Returns:
        bottleneck 배열 리스트, 일치하는 ground truth 또는 파일 이름
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []

    if how_many >= 0:
        # bottleneck 랜덤 샘플 검색
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

    else:
        # 모든 bottleneck 검색
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)
                bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, image_dir, category, bottleneck_dir,
                                                    jpeg_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor, architecture)
                bottlenecks.append(bottleneck)
                ground_truths.append(label_index)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames

#######################################################################################################################
def save_graph_to_file(sess, graph, graph_file_name):
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), [FINAL_TENSOR_NAME])
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    return

#######################################################################################################################
if __name__ == '__main__':
    main()
