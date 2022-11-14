import unittest
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) #상위 폴더 디렉토리 받아옴
from func.crop_embedding import img_embedding #img_embedding 함수 받아옴

class TestCase(unittest.TestCase):
		def setUp(self):
			self.txt_dir = "face_rec/images/faces.txt" #txt파일 디렉토리

		#txt파일에 있는 url주소가 2개 이상인지 테스트
		def test_url_num(self):
			# Given
			path = self.txt_dir

			# When 
			'''faces.txt 에서 url리스트를 가져온다''' 
			with open(path, "r") as f:
				url_num = f.readlines()
			f.close()

			# Then 
			'''txt파일의 주소가 2개 이상이어야 한다'''
			self.assertGreater(len(url_num), 1)

		#임베딩한 이미지의 개수가 2개 이상인지 테스트
		def test_dic_num(self):
			#Given
			path = self.txt_dir

			#When 
			dic_num = img_embedding(path)

			#Then
			'''딕셔너리의 갯수가 2개 이상이어야 한다.'''
			self.assertGreater(len(dic_num), 1)

if __name__ == '__main__':
    unittest.main()