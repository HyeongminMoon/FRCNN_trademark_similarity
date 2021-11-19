from scipy.stats import wasserstein_distance
#from imageio import imread
import cv2
from cv2 import imread
import numpy as np
import os
import pandas as pd

def get_histogram(img):
    h, w = img.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
    return np.array(hist) / (h * w)


if __name__ == '__main__':
    a = imread('image/big_img_4379.jpg', cv2.IMREAD_GRAYSCALE)
    a_hist = get_histogram(a)
    num_of_img = 2000
    dists = np.ones(num_of_img)
    for i in range(num_of_img):
        path = 'image/big_img_'+str(i)+'.jpg'
        if os.path.isfile(path):
            b = imread(path,cv2.IMREAD_GRAYSCALE)

            b_hist = get_histogram(b)
            dist = wasserstein_distance(a_hist,b_hist)
            #print("img"+str(i)+": "+str(dist))
            dists[i] = dist
    dist_df = pd.DataFrame(dists, columns = ['wass_dist'])
    dist_df['rank'] = dist_df['wass_dist'].rank(method='first',ascending=True)
    dist_df.sort_values(by='rank',inplace=True)
    print(dist_df)
