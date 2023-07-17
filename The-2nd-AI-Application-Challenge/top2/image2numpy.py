import cv2
import numpy as np
import os
from tqdm import tqdm

import pickle 

# 图片路径
imgpath = "../test"
imgfiles = os.listdir(imgpath)
def preprocess(img):
    img = cv2.resize(img, [224,224],interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    img_data = np.array(img)
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, 0)
    # mean_vec = np.array([123.675,116.28,103.53]).astype('float32')
    # stddev_vec = np.array([0.0171,0.0175,0.0174]).astype('float32')
    mean_vec = np.array([127.5,127.5,127.5]).astype('float32')
    stddev_vec = np.array([0.00784313725490196,0.00784313725490196,0.00784313725490196]).astype('float32')
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:, i, :, :] = (
            img_data[:, i, :, :] - mean_vec[i]) * stddev_vec[i]
    return norm_img_data

#imgs = []
#names = []
#dic = {}
#dic1 = {}
#dic2 = {}

with open("name.dat","wb")as f1:
    with open("data.dat","wb")as f2:
        for file in tqdm(imgfiles):
            input_path = imgpath + "/" + file
            origin_img = cv2.imread(input_path)
            #img = np.array(origin_img, dtype=np.float32)
            input_name = 'test/' + file
        #dic[input_name] = img
        #import pdb;pdb.set_trace()
        #names.append(input_name)
            pickle.dump(input_name, f1)
            img = preprocess(origin_img)
            img = np.array(img, dtype=np.float32)
            pickle.dump(img, f2) 
        #imgs.append(img)

#result = [input_name, imgs]
#result = numpy.rec.fromarrays((names, imgs), names=('keys', 'data'))
#imgs = np.array(imgs, dtype=np.float32)
#names = np.array(names)
#print(img.shape)
#print(names.shape)
#result = np.rec.fromarrays((names, imgs), names=('keys', 'data'))
#np.save('image2numpy_name.npy', names)
#np.save('image2numpy_data.npy', imgs)
