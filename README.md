2022.07.18 ~ 2022.08.05

[배경]

- 시니어를 위한 데이팅 앱을 개발중에 있는 기업과의 협업
- 안면인식과 챗봇 팀 중 **안면인식** 팀에서 프로젝트를 진행 (약 3주 간 프로젝트)

[목적]

- 프로필로 사용할 사진을 본인의 얼굴이 맞는지 확인한다.

[역할]

base code 작성 , 코드 총괄

[목표]

- 기존 안면 인식 모델의 불필요한 모델과 코드를 삭제하여 간결화한다.
- 얼굴 인식 모델 중 가장 빠르고 정확한 모델을 찾아 사용한다.
- 인식이 안되는 상황을 고려하여 얼굴 인식 모델을 추가한다.
- 프로필에 올릴 사진와 현재 사진의 벡터 거리 값 차(L2 norm)를 이용하여 정확도 80% 이상 동일인을 구별한다.

[진행]

- 얼굴 인식 **모델 선택**

얼굴 인식 모델 API를 활용하였다.
얼굴 인식 모델까지 만드려면 얼굴 데이터 수집부터 많은 시간이 걸리고 그 만큼 효과를 기대하기 어렵기 때문에 8가지의 모델 비교를 통해 가장 빠르고 성능이 좋은 모델을 사용하였다.

![AI_11__AI_12__CS2_](https://user-images.githubusercontent.com/87513112/201985381-4fb801ba-87c9-482d-9cde-c2313b2f3ebb.jpg)


- 얼굴 인식과 이미지 변형, 임베딩 벡터 구하는 기능이 구현되있는 dlib과 다른 얼굴 인식 모델 CaffeNet 사용
1. 68개의 얼굴 랜드마크를 이용해 이미지 전처리 후 128개의 임베딩 백터 거리 값 계산하여 동일인 비교

![Untitled (4)](https://user-images.githubusercontent.com/87513112/201985408-2c723813-7dea-4973-a463-fc99a7dab86f.png)

1. 또 다른 얼굴 인식 모델인 CaffeNet를 사용하여 dlib의 보조 역할을 한다.
2. L2 norm으로 이미지 벡터 거리 값을 계산하여 0 ~ 1 사이의 score로 나타냄

```python
embedding = np.linalg.norm(all_img_embedding[i]-all_img_embedding[self_img_name], ord=2) *# self_img_name --> 현재 사진*
```

****

[결과]

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1bebda2d-adae-436c-b497-2a104594764c/Untitled.png)

- 동일인 비교 정확도 80% 이상

*임베딩 차 score 0.4 미만는 동일인으로 판정*

*임베딩 차 score 0.4 이상은 비동일인으로 판정*

- 구동 시간 단축

로컬 **약 12초 이상 —>** **약 3초**

서버 **약 9초 이상** —> **약 2초**

[개선]

1. 이미지 데이터 전처리를 미리 진행하여 얼굴 인식 모델의 구동 시간 단축과 얼굴 인식률을 높임
2. 되도록 정면을 보고있는 사진을 사용한다.
3. 640×480 이상 해상도의 사진, 어플로 보정되지 않은 사진을 사용한다.
4. 구동시간을 줄이기위해 코드를 간단하게 만들어야한다.
5. 현존하는 인식 모델을 가져와 사용하는 것이 훨씬 효율적이다.
6. 적외선 카메라, 적외선 투광기, 근접 센서, 조도 센서, 도트 프로젝터 등 여러 센서와 같이 사용 된다면 훨씬 높은 성능을 기대할 수 있다.








```python
```
├─ func   
│    ├─ comparison.py   
│    ├─ crop_embedding.py   
│    └─ __init__.py   
├─ images   
│    └─ faces.txt   
├─ main.py   
   
```
```

### crop_embedding.py

- img_embedding
face.txt 에 있는 이미지 URL을 가져와 embedding 값을 구하여 반환하는 함수

1. *인코딩 값을 구하기 전, 속도 개선을 위해 먼저* detectAndDisplay함수로 *이미지를 crop한다.*
2. crop한 이미지를 get_face_embedding_dict *함수 안에서 임베딩 값을 구한다.*
3. detectAndDisplay 함수에서 얼굴 인식과 crop에 실패한 경우 
dlib으로 얼굴 인식과 crop을 전부 하고 이미지의 임베딩 값을 구한다.
4. 임베딩 값들을 딕셔너리 구조로 반환, comparison함수에서 사용

- detectAndDisplay
1. CaffeNet 모델을 사용하여 얼굴 인식을 하고 crop이 완료되면 이미지을 반환,
2.  img_embedding 함수에서 사용

- get_face_embedding_dict
1. dlib의 face_recognition.face_encodings 라이브러리를 사용해 이미지 임베딩 값을 반환,
2. img_embedding 함수에서 사용

CaffeNet ⇒ 얼굴 인식

dlib ⇒ 얼굴 인식, crop, 임베딩 값

### comparison.py

faces.txt 의 맨 처음 url은 현재 사진, 그 뒤 3개 는 프로필에 등록할 사진

```python
embedding = np.linalg.norm(all_img_embedding[i]-all_img_embedding[self_img_name], ord=2) *# self_img_name --> 현재 사진*
```

L2.norm 벡터 거리값 계산을 통해 

*임베딩 차 0.4 이하는 동일인 이상은 비 동일인으로 판정*







<추가한 코드></br>
--crop_embedding.py--</br>
94-95줄 : txt파일에서 불러온 이미지가 한 장 이하이면 프로그램 종료</br>
98줄 : strip()함수 추가 -> 개행문자('\n') 없앰</br>
100줄 : 확장자를 리스트 타입으로 변환</br>
101줄 : 올바른 확장자 일 때만 faces_url_list에 추가 </br>
106줄 : num == 0 , 즉 셀카의 확장자가 이상하면 프로그램 종료(더 진행할 의미가 없다)</br>
122줄 : numbering == 0 , 즉 셀카의 이미지를 불러올 수 없으면 프로그램 종료</br>
125줄 : image를 못 받으면 None을 대입</br>
128줄 : image가 None이 아닐때만 crop을 진행한다.</br>
136-137줄 : 임베딩 값을 구했을 때만 img_dict에 삽입</br>
143-144줄 : 임베딩 값을 구했을 때만 img_dict에 삽입</br>
152-153줄 : 인식한 이미지가 한 장 이하라면 프로그램 종료</br>
</br>
<테스트 코드></br>
--test_face_rec.py--</br>
1.test_url_num() : txt파일에 있는 url주소가 2개 이상인지 테스트</br>
2.test_dic_num() : 임베딩한 이미지의 개수가 2개 이상인지 테스트</br>
</br>
<삭제한 코드></br>
get_face_embedding() 함수 : </br>
짧아서 따로 함수를 만들 필요가 없다고 판단</br>
