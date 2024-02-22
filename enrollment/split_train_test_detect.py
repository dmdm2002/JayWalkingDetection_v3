import os
import re
import math
import random
import numpy as np
import warnings
import shutil
warnings.filterwarnings("ignore")

random.seed(1004)

root = 'D:/Side/2024_Sejoong_Jaywalking/DB/detection/'
img_names = os.listdir(f'{root}/images-masked')

train_cctvs = ['0008.jpg', '0038.jpg', '0007.jpg', '0005.jpg', '0020.jpg', '0037.jpg', '0010.jpg', '0043.jpg', '0028.jpg']
test_cctvs = ['0009.jpg', '0036.jpg', '0045.jpg', '06', '0016.jpg']

# test --> 0009.jpg(5), 0036.jpg(2), 0045.jpg(2), 06(11), 0016.jpg(14)
train_root = 'D:/Side/2024_Sejoong_Jaywalking/DB/detection/train-masked'
os.makedirs(f'{train_root}/images', exist_ok=True)
os.makedirs(f'{train_root}/labels-1', exist_ok=True)
os.makedirs(f'{train_root}/labels-2', exist_ok=True)

test_root = 'D:/Side/2024_Sejoong_Jaywalking/DB/detection/test-masked'
os.makedirs(f'{test_root}/images', exist_ok=True)
os.makedirs(f'{test_root}/labels-1', exist_ok=True)
os.makedirs(f'{test_root}/labels-2', exist_ok=True)

for name in img_names:
    cctv = name.split('-')[1]
    ann = re.compile('.jpg').sub('.txt', name)
    if cctv in train_cctvs:
        shutil.copyfile(f'{root}/images-masked/{name}', f'{train_root}/images/{name}')
        shutil.copyfile(f'{root}/labels-1/{ann}', f'{train_root}/labels-1/{ann}')
        shutil.copyfile(f'{root}/labels-2/{ann}', f'{train_root}/labels-2/{ann}')

    else:
        shutil.copyfile(f'{root}/images-masked/{name}', f'{test_root}/images/{name}')
        shutil.copyfile(f'{root}/labels-1/{ann}', f'{test_root}/labels-1/{ann}')
        shutil.copyfile(f'{root}/labels-2/{ann}', f'{test_root}/labels-2/{ann}')


# cctv_id = []
# for name in img_names:
#     cctv = name.split('-')[1]
#     cctv_id.append(cctv)
#
# cctv_names = list(set(cctv_id))
# print(cctv_names)
#
# unique, count = np.unique(cctv_id, return_counts=True)
# print("리스트 내 나열 순")
# for u, c in zip(unique, count):
#     print(u +"("+str(c)+")", end=" ")
