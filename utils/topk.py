import torch
import numpy as np
import os
import shutil
from numpy import linalg as LA
from numpy import dot
def cos_sim(A, B):
    return dot(A, B) / (LA.norm(A) * LA.norm(B))

if __name__ == '__main__':
    input_id = 9
    npys_path = '../../npys/'
    npys = list(sorted(os.listdir(npys_path)))
     
    input_feature = np.load(npys_path + str(input_id) + ".npy")
    output = []
    for n in npys:
        load_feature = np.load(npys_path + n)
        cos = cos_sim(input_feature[0], load_feature[0])
        if cos > 0.8:
            output.append(n)
    print(len(output))
    img_path = '../../images/'
    for o in output:
        shutil.copy(img_path + "img_" + o[:-4] + ".jpg", "topk/img_" + o[:-4] + ".jpg")
