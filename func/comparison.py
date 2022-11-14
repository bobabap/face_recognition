"""
 얼굴 인증 메인 함수
 실행 시간 : 2.137037992477417
"""
import cv2
import numpy as np
import face_recognition
import urllib.request 
import sys 
import time
# start = time.time()
# print("time :", time.time() - start)

from func.crop_embedding import img_embedding


def comparison():
    '''#인자값 : img주소가 있는 txt파일 디렉토리 #리턴값 : key = 숫자이름, value = 임베딩값의 딕셔너리'''
    all_img_embedding = img_embedding("face_rec/images/faces.txt") 

    '''faces.txt 의 맨 처음 url은 현재 사진이다'''
    all_img_name = [i for i in all_img_embedding.keys()] # 숫자 이름만 리스트로
    self_img_name = all_img_name[0] # 현재 사진 이름
    profile_photo_name = all_img_name[1:] # 프로필에 등록하려는 사진이름 


    all_img = {} # 전체 프로필 사진
    for i in profile_photo_name: 
        '''딕셔너리 key = ("0","1","2"...)값를 이용해 value를 가져오고 셀피와 embedding 차이를 구함'''
        embedding = np.linalg.norm(all_img_embedding[i]-all_img_embedding[self_img_name], ord=2) # self_img_name --> 현재 사진
        all_img[i] = round(embedding, 3) # 소수점 3자리까지 딕셔너리에 key(프로필사진번호) : value(거리값)형태로 저장
    
    allowed_photo = {} # 사용 가능
    disallowed_photo = {} # 사용 불가능

    for tup in all_img.items():
        '''tup[1]은 거리값 / tup[0]은 사진이름번호'''
        if tup[1] <= 0.4: # 임베딩 차 0.4 이하는 동일인 이상은 비동일인
            allowed_photo[tup[0]] = round(tup[1], 3) # 동일인에 저장
        else:
            disallowed_photo[tup[0]] = round(tup[1], 3) # 비동일인에 저장
    
    print('등록 하려는 프로필 사진 :\n', all_img, 'count :', len(all_img))
    print('-------------------')
    print('비동일인 판정 60% > :\n', disallowed_photo)
    print('동일인 판정 60% < :\n', allowed_photo)
    print('-------------------')
    
    if len(allowed_photo) >= 1: # 현재와 가장 동일한 사진을 하나 올리면 나머지는 자유
        print('프로필 등록이 완료되었습니다.')
    else:
        print(f"닮은 사진이 없습니다. 다시 시도하세요")
