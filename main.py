from func.comparison import comparison
from func.crop_embedding import img_embedding
import time

 
if __name__ == '__main__':
    start = time.time()
    
    comparison() # 임베딩 값으로 사진 비교

    print("time :", time.time() - start)