import os
import tensorflow as tf
import numpy as np
import cv2

RETRAINED_LABELS_TXT_FILE_LOC = os.getcwd() + "/" + "retrained_labels.txt"
RETRAINED_GRAPH_PB_FILE_LOC = os.getcwd() + "/" + "retrained_graph.pb"

TEST_IMAGES_DIR = os.getcwd() + "/test_images"

SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLUE = (255.0, 0.0, 0.0)

#######################################################################################################################
def main():

    print("프로그램 시작...")

    # 테스트 파일들이 존재하는지 확인
    if not checkIfNecessaryPathsAndFilesExist():
        return

    # 레이블 파일에서 분류 목록을 가져오기
    classifications = []
    # 레이블 파일의 각 분류 목록 확인 후 반복
    for currentLine in tf.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
        # 캐리지 리턴을 제거
        classification = currentLine.rstrip()
        # 리스트 추가하기
        classifications.append(classification)

    # 레이블 파일 읽은 것 프린트
    print("분류목록 = " + str(classifications))

    # 파일에서 그래프 로드
    with tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
        # GraphDef 객체 인스턴스화
        graphDef = tf.GraphDef()
        # 학습된 그래프로 GraphDef 객체 읽기
        graphDef.ParseFromString(retrainedGraphFile.read())
        # 그래프를 현재 기본 Graph로 가져오기
        _ = tf.import_graph_def(graphDef, name='')

    # 위에 나열된 테스트 이미지 디렉터리가 올바르지 않으면 오류 메시지를 표시하고 리턴
    if not os.path.isdir(TEST_IMAGES_DIR):
        print("테스트 이미지 디렉토리가 올바르지 않습니다.")
        print("파일/디렉토리 경로를 확인해 주세요.")
        return

    with tf.Session() as sess:
        # 테스트 이미지 디렉토리의 각 파일에 대해 반복
        for fileName in os.listdir(TEST_IMAGES_DIR):
            # 파일이 .jpg 또는 .jpeg로 끝나지 않는 경우(대/소문자 구분 안 함) 넘어가기
            if not (fileName.lower().endswith(".jpg") or fileName.lower().endswith(".jpeg")):
                continue

            # 파일 이름 표시
            print(fileName)

            # 현재 이미지 파일의 전체 경로 및 파일 이름 가져오기
            imageFileWithPath = os.path.join(TEST_IMAGES_DIR, fileName)
            # OpenCV로 영상 열기 시도
            openCVImage = cv2.imread(imageFileWithPath)

            # 이미지를 열지 못할 경우
            if openCVImage is None:
                print("파일 " + fileName + " 을 OpenCV로 열 수 없습니다.")
                continue

            # 그래프에서 최종 텐서 얻기
            finalTensor = sess.graph.get_tensor_by_name('final_result:0')

            # OpenCV 영상(numpy array)을 TensorFlow 이미지로 변환
            tfImage = np.array(openCVImage)[:, :, 0:3]
            
            # 예측값 얻기 위해 네트워크 가동
            predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})

            # 예측값 높은 것을 우선으로 해서 분류
            sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

            print("---------------------------------------")

            # 예측에 대한 자세한 정보를 표시할 수 있도록 아래 루프를 통과하는지 확인(위에서 아래로 정렬)
            onMostLikelyPrediction = True
            # 예측 표시
            for prediction in sortedPredictions:
                strClassification = classifications[prediction]

                # 분류(디렉토리 이름에서 확인됨)가 문자 "s"로 끝나는 경우 "s"를 제거하여 복수에서 단수로 변경
                if strClassification.endswith("s"):
                    strClassification = strClassification[:-1]

                # 다음 소수점 뒤에 두 자리로 반올림
                confidence = predictions[0][prediction]

                # 가장 예측이 높은 케이스를 통해 어떤 것으로 예측되는지 설명하고 소수점 두 자리까지 % 확률를 표시
                if onMostLikelyPrediction:
                    # % 로 표시
                    scoreAsAPercent = confidence * 100.0
                    # 결과 리턴
                    print("예측 :  " + strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% 확률")
                    # 결과 값 이미지에 보여주기
                    writeResultOnImage(openCVImage, strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence")
                    # OpenCV 이미지를 보여주기
                    cv2.imshow(fileName, openCVImage)
                    # 가장 확률이 높은 예측을 표시. 
                    onMostLikelyPrediction = False

                # 예측을 위해 소수점 5자리까지 확률을 표시
                print(strClassification + " (" +  "{0:.5f}".format(confidence) + ")")

            # pause until a key is pressed so the user can see the current image (shown above) and the prediction info
            cv2.waitKey()
            # after a key is pressed, close the current window to prep for the next time around
            cv2.destroyAllWindows()

    # TensorBoard로 볼 수 있도록 그래프를 파일에 기록하십시오.
    tfFileWriter = tf.summary.FileWriter(os.getcwd())
    tfFileWriter.add_graph(sess.graph)
    tfFileWriter.close()

#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(TEST_IMAGES_DIR):
        print('')
        print('에러: TEST_IMAGES_DIR "' + TEST_IMAGES_DIR + '" 파일이 존재하지 않습니다.')
        print('테스트 이미지를 확인 해 주세요 ')
        print('')
        return False

    if not os.path.exists(RETRAINED_LABELS_TXT_FILE_LOC):
        print('에러: RETRAINED_LABELS_TXT_FILE_LOC "' + RETRAINED_LABELS_TXT_FILE_LOC + '" 파일이 존재하지 않습니다.')
        return False

    if not os.path.exists(RETRAINED_GRAPH_PB_FILE_LOC):
        print('에러: RETRAINED_GRAPH_PB_FILE_LOC "' + RETRAINED_GRAPH_PB_FILE_LOC + '" 파일이 존재하지 않습니다.')
        return False

    return True

#######################################################################################################################
def writeResultOnImage(openCVImage, resultText):

    imageHeight, imageWidth, sceneNumChannels = openCVImage.shape

    # 폰트 선택
    fontFace = cv2.FONT_HERSHEY_TRIPLEX

    # 글꼴 크기와 두께를 선택
    fontScale = 1.0
    fontThickness = 2

    # 글꼴 두께가 정수인지 확인. 정수가 아니면 OpenCV 기능과 충돌할 수 있음
    fontThickness = int(fontThickness)

    upperLeftTextOriginX = int(imageWidth * 0.05)
    upperLeftTextOriginY = int(imageHeight * 0.05)

    textSize, baseline = cv2.getTextSize(resultText, fontFace, fontScale, fontThickness)
    textSizeWidth, textSizeHeight = textSize

    # 텍스트 영역 중심, 너비 및 높이를 기준으로 텍스트 영역의 왼쪽 아래 원점을 계산
    lowerLeftTextOriginX = upperLeftTextOriginX
    lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight

    # 이미지에 글 넣기
    cv2.putText(openCVImage, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, SCALAR_BLUE, fontThickness)

#######################################################################################################################
if __name__ == "__main__":
    main()
