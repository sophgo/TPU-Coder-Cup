#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import numpy as np
import argparse
from tpu_perf.infer import SGInfer
import cv2
import linecache
from tools.model_runner import mlir_inference, onnx_inference, caffe_inference, model_inference
# import pdb
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
    # pdb.set_trace() 
    return norm_img_data


def parse_args():
    parser = argparse.ArgumentParser(description='Inference xception network.')
    parser.add_argument("--model_def", type=str,
                        required=True, help="Model definition file")
    parser.add_argument("--model_data", type=str,
                        help="Caffemodel data, only for caffe model")
    parser.add_argument("--net_input_dims", type=str,
                        default="224,224", help="(h,w) of net input")
    parser.add_argument("--input", type=str, required=True,
                        help="Input image for testing")
    parser.add_argument("--output", type=str, required=True,
                        help="Output image after classification")
    parser.add_argument("--category_file", type=str, required=True,
                        help="The index file of 1000 object categories")
    args = parser.parse_args()
    return args


def postprocess(output, img, category_file, top_k=1):
    #prob = output['resnetv22_dense0_fwd_Gemm'] # according to the network output
    #import pdb; pdb.set_trace()
    #if('output_Gemm_f32' in output.keys()):
    #    prob = output['output_Gemm_f32']
    #if('output_Gemm' in output.keys()):
     #   prob = output['output_Gemm']
    #else:
    #prob = output[1]
    return np.argmax(output[1][0].flatten())
 
    '''txt_bk_color = (0, 0, 0)
    txt_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_txt = []
    line_txt.append('Top-{:d}'.format(top_k))
    txt_size = cv2.getTextSize(line_txt[0], font, 0.4, 1)[0]
     text left_bottom
    txt_x0 = 1
    txt_y0 = txt_size[1]
     rectangle left top
    rect_x0 = 0
    rect_y0 = 0
     rectangle right bottom
    rect_x1 = txt_size[0] + 2
    rect_y1 = int(1.5 * txt_size[1])
    step = int(1.5 * txt_size[1])
    img = cv2.rectangle(img, (rect_x0, rect_y0),
                        (rect_x1, rect_y1), txt_bk_color, thickness=-1)
    img = cv2.putText(
        img, line_txt[0], (txt_x0,  txt_y0), font, 0.4, txt_color, thickness=1)'''
    '''
    
    line_txt.append(linecache.getline(category_file, top_k_idx[i]+1).strip('\n'))
        #txt_size = cv2.getTextSize(line_txt[i + 1], font, 0.4, 1)[0]
        img = cv2.rectangle(img, (0, rect_y1 + i * step),
                            (txt_size[0] + 2, rect_y1 + (i + 1) * step), txt_bk_color, thickness=-1)
        img = cv2.putText(
            img, line_txt[i + 1], (1,  txt_y0 + (i + 1) * step), font, 0.4, txt_color, thickness=1)
        i += 1'''
    #return top_k_idx


def main():
    import pandas
    import os
    #import time
    #t1=time.time()

    args = parse_args()
    output_path=args.output
    lst=[]
    lst1=[]
    input_shape = tuple(map(int, args.net_input_dims.split(',')))
    import json
    #model_tmp = arg.model_def
    with open('./config.json') as f:
        dataset = json.load(f)
    # print(data['id2label']['1'])
    dic={"file_name":[],"label":[]}
    model = SGInfer("vit_1684x_INT8.bmodel",devices=[0])
    for j in range(1,8580):
        i = "./test_dataset/"+str(j) + '.jpg'
        filename=os.path.join(args.input,i)
        origin_img = cv2.imread(i)  # RGB HWC
        img = preprocess(origin_img, input_shape)
        data = {'data': img}  # input name from model
        output = dict()
        #3if args.model_def.endswith('.prototxt') and args.model_data.endswith('.caffemodel'):
        
          #  output = caffe_inference(data, args.model_def, args.model_data, False)
        #elif args.model_def.endswith('.mlir'):
        #from tpu_perf.infer import SGInfer
        #model = SGInfer("vit_1684x_INT8.bmodel",devices=[0])
        #output =  model.put(arr)
        #output = model.get()i

         #   output = mlir_inference(data, args.model_def, False)
        #elif args.model_def.endswith(".bmodel"):
        #import pdb;pdb.set_trace()
        output =  model.put(img)
        output = model.get()


        #output = model_inference(data, model_tmp)
        #elif args.model_def.endswith(".onnx"):
        #from tpu_perf.infer import SGInfer
        #model = SGInfer("vit_1684x_INT8.bmodel",devices=[0])
        #output =  model.put(arr)
        #output = model.get()
        
       #     output = onnx_inference(data, args.model_def)
        
        #else:
         #   raise RuntimeError("not support modle file:{}".format(args.model_def))
        fix_img = postprocess(output, origin_img, args.category_file)
        #cv2.imwrite(args.output, fix_img
        #lst1.append(i)
        #rr=dataset['id2label'][str(fix_img)].split(",")[0]
        lst.append(dataset['id2label'][str(fix_img)].split(",")[0])
    #import pdb;pdb.set_trace()
    output1=pandas.DataFrame(lst,columns = None)
    #output2=pandas.DataFrame(lst1)
    #output2.columns = ['file_name']
    #output2.insert(output2.shape[1], columns=['label'], output1) 
    #output2 = pandas.concat([output2, pandas.DataFrame(lst, columns = ['label'])], sort=False)
    # for kk in range(5): 
    #output2 = output2.reindex(columns=['file_name','label'],fill_value=lst[4])
    output1.to_csv(output_path,index=False,header=True)
    #t2=time.time()

    #print((t2-t1)*1000)
    #output1.to_csv(output_path,mode = 'a',index=False,header=True)
if __name__ == '__main__':
    main()

