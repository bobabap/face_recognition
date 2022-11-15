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

- face.txt 에 있는 이미지 URL을 가져와 embedding 값을 구하여 반환하는 함수

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
