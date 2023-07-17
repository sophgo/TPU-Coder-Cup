#final submission
import os
import pandas as pd
import numpy as np
import argparse
import cv2
import linecache
from tpu_perf.infer import SGInfer
from tools.model_runner import mlir_inference, model_inference, onnx_inference, caffe_inference
from tqdm import tqdm
from time import time

# The sample for resnet18_v2

def preprocess(img, input_shape):
    img = cv2.resize(img, input_shape,
                     interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    img_data = np.array(img)
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, 0)
    mean_vec = np.array([127.5,127.5,127.5]).astype('float32')
    stddev_vec = np.array([0.00784313725490196,0.00784313725490196,0.00784313725490196]).astype('float32')
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
    parser.add_argument("--input", type=str, required=True,
                        help="Input image for testing")
    parser.add_argument("--output", type=str, required=True,
                        help="Output image after classification")
    parser.add_argument("--category_file", type=str, required=True,
                        help="The index file of 1000 object categories")
    args = parser.parse_args()
    return args


def postprocess(output, img, category_file, top_k=5):
    #import pdb;pdb.set_trace()
    index = np.argmax(output)
    return index
	
def main():
    args = parse_args()
    input_shape = tuple(map(int, args.net_input_dims.split(',')))
    #import pdb;pdb.set_trace()
    dic = {'file_name':[], 'label':[]}
	model = SGInfer("vit_1684x_f32.bmodel", devices=[0])
    #pipeline = Pipeline()
    #id2label = np.load("id2label.npy", allow_pickle=True).item()
    import json 
    with open('config.json', 'r') as file:
        id2label = json.load(file)  #loads the configuration config.json which contains dictionaries of {class : labels} into a numpy array, id2label
        
    preprocess_time = 0
    calculate_time = 0
    postprocess_time = 0
    for img_name in tqdm(os.listdir(args.input)):
        begin_time = time()
        file_name = os.path.join(args.input, img_name)
        dic['file_name'].append(file_name)

        origin_img = cv2.imread(file_name)
        img = preprocess(origin_img, input_shape)
        data = {'data': img}  # input name from model
        output = dict()

        t1 = time()
        preprocess_time += t1 - begin_time
        t1 = time()
        #output = pipeline.forward(data['data'])
        #output = output[1]

        
        #preprocess_time += t1 - begin_time
        if args.model_def.endswith('.prototxt') and args.model_data.endswith('.caffemodel'):
            output = caffe_inference(data, args.model_def, args.model_data, False)
        elif args.model_def.endswith('.mlir'):
            output = mlir_inference(data, args.model_def, False)
        elif args.model_def.endswith(".bmodel"):
			output = model.put(data['data'])
            output = model.get()
            #output = pipeline.forward(data['data'])
            #output = pipeline.forward(data['data'])
            output = output[1]
            t2 = time()
            calculate_time += t2 - t1
            #task_id, output = model.get()
            #import pdb;pdb.set_trace()
            #output = model_inference(data, args.model_def)
        elif args.model_def.endswith(".onnx"):
            output = onnx_inference(data, args.model_def)
        else:
            raise RuntimeError("not support modle file:{}".format(args.model_def))
        t2 = time()
        calculate_time += t2 - t1
        #index = np.argmax(output)
		line_txt = postprocess(output, origin_img, args.category_file, top_k=1)#index of the max probability of image category
        #import pdb;pdb.set_trace()
		value = id2label['id2label'].get(str(line_txt)) #returns the dictinary value of an index=line_txt
        dic['label'].append(value)
        t3 = time()
        postprocess_time += t3 - t2
        #fix_img = postprocess(output, origin_img, args.category_file, top_k=1)
        #dic['label'].append(model.config.idx2label[index])
        #cv2.imwrite(args.output, fix_img)
    df = pd.DataFrame(dic)  #convert the pandas frame
    df.to_csv('data.csv', index=False)  #saves into a csv file format
    print(f"preprocess time: {preprocess_time}")
    print(f"calculate time: {calculate_time}")
    print(f"postprocess time: {postprocess_time}")

if __name__ == '__main__':
    main() 
	