import os
from PIL import Image
import cv2
import numpy as np


def merge_txt(file_prefix):
    vienna_codes = []
    for i in range(1,51):
        path = '../python_parse_kipris_/'+file_prefix+'_'+str(i)+'.txt'
        vienna_fp = open(path,'r')
        lines = vienna_fp.readlines()
        vienna_codes.extend(lines)
        vienna_fp.close()
    fp = open('../python_parse_kipris_/'+file_prefix+'.txt','w')
    print(vienna_codes)
    for vienna_code in vienna_codes:
        fp.write(vienna_code)
    fp.close()

def classify():
    vienna_fp = open('../python_parse_kipris_/viennacode.txt','r')
    vienna_codes = vienna_fp.readlines()
    vienna_codes = np.array(vienna_codes)
    vienna = []
    for i in range(len(vienna_codes)):
        vienna_codes[i] = vienna_codes[i].replace('\n','')
        vienna.append(vienna_codes[i].split('|'))
    #print(vienna)
    for i in range(1,25001):
        path = 'image/big_img_' + str(i) + '.jpg'
        if os.path.isfile(path):
            img = Image.open(path)
            temp_set = set()
            for v in vienna[i-1]:
                temp_set.add(v[0:6])
            for t in temp_set:
                if not os.path.isdir('image/'+t):
                    os.mkdir('image/'+t)
                img_fp = open('image/'+t+'/big_img_'+str(i)+'.jpg','w')
                img.save(img_fp,"JPEG")
    vienna_fp.close()



if __name__ == '__main__':
    #merge_txt('viennacode')
    Image.LOAD_TRUNCATED_IMAGES = True
    classify()

