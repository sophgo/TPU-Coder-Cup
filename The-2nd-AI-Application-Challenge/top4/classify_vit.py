#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import numpy as np
import argparse
import cv2
# import linecache
# from tools.model_runner import mlir_inference, model_inference, onnx_inference, caffe_inference
# from tpu_perf.infer import SGInfer
import numpy as np
# from tqdm import tqdm
# from time import time

# The sample for resnet18_v2

def preprocess(img, input_shape):
    img = cv2.resize(img, input_shape,
                     interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    img_data = np.array(img)
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, 0)
    mean_vec = np.array([123.675,116.28,103.53]).astype('float32')
    stddev_vec = np.array([0.0171,0.0175,0.0174]).astype('float32')
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:, i, :, :] = (
            img_data[:, i, :, :] - mean_vec[i]) * stddev_vec[i]
   
    return norm_img_data


def parse_args():
    parser = argparse.ArgumentParser(description='Inference xception network.')
    parser.add_argument("--model_def", type=str,
                        required=True, help="Model definition file")
    parser.add_argument("--model_data", type=str,
                        help="Caffemodel data, only for caffe model")
    parser.add_argument("--net_input_dims", type=str,
                        default="224,224", help="(h,w) of net input")
    # parser.add_argument("--input", type=str, required=True,
    #                     help="Input image for testing")
    # parser.add_argument("--output", type=str, required=True,
    #                     help="Output image after classification")
    # parser.add_argument("--category_file", type=str, required=True,
    #                     help="The index file of 1000 object categories")
    args = parser.parse_args()
    return args

def model_inference(inputs, model):
    output = model.put(inputs)
    output = model.get()
    return output



def main():
    import pandas as pd
    import os
    from npy_append_array import NpyAppendArray
    args = parse_args()
    input_shape = tuple(map(int, args.net_input_dims.split(',')))

    # 加载模型
    model = SGInfer("vit_1684x_f32.bmodel", devices=[0])
    id2label = np.load("id2label.npy", allow_pickle=True).item()

    # 预先对图片进行预处理，并将图片保存为Numpy
    '''
    origin_img = cv2.imread('test/1.jpg')
    img = preprocess(origin_img, input_shape)
    
    data = np.array(img)
    print(data.shape)
    np.save('data.npy', data)
    data = NpyAppendArray('data.npy')
    for i in range(1, 8580):
        img_name = 'test/{}.jpg'.format(i + 1)
        origin_img = cv2.imread(img_name)
        img = preprocess(origin_img, input_shape)
        data.append(img)
    '''
    
    
    data = np.load('data.npy', allow_pickle=True)


    # img = preprocess(origin_img, input_shape)
    # data = {'data': img}  # input name from model
    dct = dict()
    if args.model_def.endswith('.prototxt') and args.model_data.endswith('.caffemodel'):
        # output = caffe_inference(data, args.model_def, args.model_data, False)
        pass
    elif args.model_def.endswith('.mlir'):
        # output = mlir_inference(data, args.model_def, False)
        pass
    elif args.model_def.endswith(".bmodel"):
        file_name = []
        label_id = []
        d_len = data.shape[0]
        for i, inputs in enumerate(data):
            output = model_inference(inputs.reshape(-1,3, 224, 224), model)[1]
            label_id.append(np.argmax(output))
        for i in range(d_len):
            file_name.append('test/{}.jpg'.format(i))
        label = id2label[label_id]
        file_name = np.array(file_name)
        dct['file_name'] = file_name
        dct['label'] = label
    elif args.model_def.endswith(".onnx"):
        # output = onnx_inference(data, args.model_def)
        pass
    else:
        raise RuntimeError("not support modle file:{}".format(args.model_def))
    df = pd.DataFrame(dct)
    pd.to_csv('ans.csv', df, index=False)


if __name__ == '__main__':
    main()
