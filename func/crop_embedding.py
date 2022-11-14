'''
find face and crop and embedding

실행 시간 : 0.5425937175750732
''' 
import cv2
import numpy as np
import face_recognition
import time
import sys 
# start = time.time()
# print("time :", time.time() - start)
import requests

'''얼굴 이미지를 임베딩값으로 변환'''
def get_face_embedding_dict(numbering, url, img): # numbering = img번호, url = 사진주소, img = 이미지 자체

    embedding = face_recognition.face_encodings(img)   # 얼굴 영역에서 얼굴 임베딩 벡터를 추출
    
    if len(embedding) > 0:   # crop한 이미지에서 얼굴 영역이 제대로 detect되지 않았을 경우를 대비
        '''crop한 사진에서 임베딩값을 구한경우 통과'''
        return embedding[0] # 임베딩 값 리턴
    else:
        '''crop한 이미지가 임베딩 값을 구하지 못할 때 원본 사진으로 임베딩값을 구한다.'''
        image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
        image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)

        embedding = face_recognition.face_encodings(image)

        if len(embedding) > 0:
            pass

        else:
            '''원본 사진마저 인식을 못하는 경우'''
            # 나중에 다른모델로 detection할 수 있음
            
            if numbering == 0: # 셀카를 인식 못하면 프로그램 종료
                sys.exit('셀카를 인식할 수 없습니다')

            else : #셀카가 아니면 계속 진행     
                print(url, '가 아닌 다른 사진을 넣어주세요. --> 얼굴 인식 안됌') 

            return  '0' #원본사진마저 인식을 못하는 url은 '0'을 리턴
 
        return embedding[0] # 임베딩 값 리턴
    
model_name='face_rec/res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name='face_rec/deploy.prototxt.txt'

def detectAndDisplay(img):
    # 원본 사진을 face_recognition.face_encodings 를 활용해
    # crop과 encoding을 할 수 있지만 시간이 오래걸리기 때문에
    # 크롭하고 인코딩값을 구한다.
    
    (height, width) = img.shape[:2]
    model=cv2.dnn.readNetFromCaffe(prototxt_name,model_name)
    blob=cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),1.0, (300,300),(104.0,177.0,123.0))
    
    model.setInput(blob)
    
    detections=model.forward()

    '''crop 과정'''
    min_confidence=0.9
    result_img = '0'
    for i in range(0, detections.shape[2]):
        
        confidence = detections[0, 0, i, 2]
        
        if confidence > min_confidence:
              
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            
            if height > endY and width > endX and startX > 0 and startY > 0: #예외처리  
                result_img = img[startY:endY,startX:endX]  
                min_confidence = confidence  #최소 얼굴인식 확률을 현재 확률로 정함            

    '''crop을 했을 경우'''
    if len(result_img) > 1:
        return result_img
    
    else:    
        '''crop 하지 못하는경우'''
        return []


def img_embedding(path): #path는 main.py에 있는 txt디렉토리
    faces_url_list = [] # url리스트
    '''faces.txt 에서 url리스트를 가져온다''' # faces.txt의 url리스트가 0이상 없으면 종료
    with open(path, "r") as f:
        data = f.readlines()
    
        if len(data) <= 1: #이미지가 한 장 이하이면 프로그램 종료
            sys.exit('이미지가 없습니다.')

        for num, line in enumerate(data):
            extension = line.split('.')[-1].strip() #확장자

            if extension in ['jpg','png','jpeg','jfif']: # 올바른 확장자인지 판별
                faces_url_list.append(line.strip())  #올바른 확장자일 때만 이미지 주소추가 수정

            else:
                print('Allow png, jpg, jpeg, jfif extensions only')
                
                if num == 0: #만약 '셀카'의 확장자가 이상하면 프로그램 종료
                    sys.exit(f'셀카의 확장자가 {extension} 입니다.')
                    
                      

    img_dict = {} # {0:[array], 1:[array], 2:[array], 3:[array], 4:[array]} value = 임베딩값
    '''위 img_dict 딕셔너리를 위해 주소가 아닌 "0", "1", "2"... numbering 으로 바꿈'''
    for numbering, url in enumerate(faces_url_list):

        try:
            '''url로 이미지 받아오기'''
            image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
            image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
         
        except:
            #셀카 이미지를 받아올 수 없으면 프로그램 종료
            if numbering == 0:
                sys.exit('셀카 이미지를 받아올 수 없습니다.')
            print('해당 url 이미지를 불러올 수 없음')
            image = None

        '''image를 잘 받았을 때 crop하기'''
        if image is not None:

            crop_img = detectAndDisplay(image)
            
            '''crop 할 수 없으면 원본 이미지로 임베딩 값 구하기'''
            if len(crop_img) == 0: # crop_img = 위 리턴값을 []로 했기 때문에 리스트 내용은 0
                original_embedded = get_face_embedding_dict(numbering, url, image) # 숫자이름, 이미지 주소, 원본 이미지
            
                if len(original_embedded) > 1 : # 임베딩 값을 구했으면 딕셔너리에 삽입         
                    img_dict[numbering] = original_embedded # key = 숫자이름, value = 임베딩값          
            
            else : 
                '''#crop이 성공했을 때 임베딩 값 구하기'''
                crop_embedded = get_face_embedding_dict(numbering, url, crop_img)
             
                if len(crop_embedded) > 1: # 임베딩 값이 구했으면 딕셔너리에 삽입
                    img_dict[numbering] = crop_embedded # key = 숫자이름, value = 임베딩값 


        '''이미지 잘 받아오는지 확인'''
        # cv2.imshow(url, image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    '''만약 프로필 사진을 한 장 이하로 인식을 했다면'''    
    if len(img_dict) <= 1:
        sys.exit('인식할 수 있는 프로필 사진이 없습니다')

    return img_dict # 딕셔너리 리턴