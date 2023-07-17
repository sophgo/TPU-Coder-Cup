#!/usr/bin/env python3
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================
import os

import numpy as np
import argparse
import cv2
import linecache
from tools.model_runner import mlir_inference, model_inference, onnx_inference, caffe_inference
from tpu_perf.infer import SGInfer

import importlib
import struct
import shutil
from utils.misc import str2bool

# from transformers import ViTImageProcessor, ViTForImageClassification
from tqdm.contrib import tzip
import time
import pandas  as pd
from tqdm import tqdm
import pickle

# The sample for resnet18_v2

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


def parse_args():
    parser = argparse.ArgumentParser(description='Inference xception network.')
    parser.add_argument("--model_def", type=str, default="vit_1684x_f32.bmodel", help="Model definition file")
    parser.add_argument("--model_data", type=str,
                        help="Caffemodel data, only for caffe model")
    parser.add_argument("--net_input_dims", type=str,
                        default="224,224", help="(h,w) of net input")
    parser.add_argument("--input", type=str, default="../test",
                        help="Input image for testing")

    args = parser.parse_args()
    return args


def postprocess(output):
    #prob = output['resnetv22_dense0_fwd_Gemm'] # according to the network output
    #if('output_Gemm_f32' in output.keys()):
    #    prob = output['output_Gemm_f32']
    #if('output_Gemm' in output.keys()):
    #    prob = output['output_Gemm']
    #if('1507_Gemm' in output.keys()):
    prob = output['1507_Gemm']
    #idx = np.where(prob==np.max(prob))
    #idx = idx[-1][0]
    idx = prob.argmax(-1).item()
    return idx

def model_inference_v2(data, model_def):
    model = SGInfer(model_def, devices=[5])
    output = model.put(data) #data float32 shape
    output = model.get()

def round_away_from_zero(x):
    a = np.floor(np.abs(x) + 0.5)
    return np.sign(x) * a


def bf16_to_fp32(d_bf16):
    s = d_bf16.shape
    d_bf16 = d_bf16.flatten()
    assert d_bf16.dtype == np.uint16
    d_fp32 = np.empty_like(d_bf16, dtype=np.float32)
    for i in range(len(d_bf16)):
        d_fp32[i] = struct.unpack('<f', struct.pack('<HH', 0, d_bf16[i]))[0]
    return d_fp32.reshape(s)


def fp32_to_bf16(d_fp32):
    s = d_fp32.shape
    d_fp32 = d_fp32.flatten()
    assert d_fp32.dtype == np.float32
    d_bf16 = np.empty_like(d_fp32, dtype=np.uint16)
    for i in range(len(d_bf16)):
        bytes = struct.pack('f', d_fp32[i])
        d_bf16[i] = struct.unpack('<H', struct.pack('BB', bytes[2], bytes[3]))[0]
    return d_bf16.reshape(s)


def show_fake_cmd(in_npz: str, model: str, out_npz: str):
    print("[CMD]: model_runner.py --input {} --model {} --output {}".format(in_npz, model, out_npz))


def get_chip_from_model(model_file: str) -> str:
    fd = os.popen("model_tool --chip {}".format(model_file))
    chip = fd.read()
    fd.close()
    return chip


def pack_bmodel_context_generator(model_file, net):
    out_dir = model_file.rsplit(".", maxsplit=1)[0]
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(model_file, os.path.join(out_dir, "compilation.bmodel"))
    shutil.copy(model_file + ".json", os.path.join(out_dir, "tensor_location.json"))
    with open(out_dir + "/input_ref_data.dat", "wb") as f:
        for i in net.inputs:
            i.data.tofile(f)
    yield
    with open(out_dir + "/output_ref_data.dat", "wb") as f:
        for o in net.outputs:
            o.data.tofile(f)

def main():
    # model_1 = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    id2label = np.load("id2label.npy", allow_pickle=True).item()
    args = parse_args()
    #input_shape = tuple(map(int, args.net_input_dims.split(',')))
    #模型定义
    pyruntime = "pyruntime_"
    #is_cv18xx = False
    model_file = args.model_def
    #if model_file.endswith(".bmodel"):
    pyruntime = pyruntime + "bm"
    chip = get_chip_from_model(model_file)
    # trick for runtime link chip cmodel
    lib_so = 'libcmodel_1684x.so'
    #import pdb;pdb.set_trace()
    #if chip == 'BM1686' or chip == 'CV186X':
    #        lib_so = 'libcmodel_1686.so'
    #    elif chip == 'BM1684':
    #        lib_so = 'libcmodel_1684.so'
    cmd = 'ln -sf $TPUC_ROOT/lib/{} $TPUC_ROOT/lib/libcmodel.so'.format(lib_so)
    os.system(cmd)
    #elif model_file.endswith(".cvimodel"):
    #    pyruntime = pyruntime + "cvi"
    #    is_cv18xx = True
    #else:
    #    raise RuntimeError("not support modle file:{}".format(model_file))
    pyruntime = importlib.import_module(pyruntime)
    
    dic = {"file_name" : [], "label" : []}
    outputs = dict()
    #import pdb;pdb.set_trace()
    #if not is_cv18xx:
    model = pyruntime.Model(model_file)
    net = model.Net(model.networks[0])
    #else:
    #    model = pyruntime.Model(model_file, output_all_tensors=dump_all)
    #    net = model

    preprocess_time = 0
    calculate_time = 0
    postprocess_time = 0
    #import pdb;pdb.set_trace()

    with open("name.dat", "rb")as f1:
        with open("data.dat", "rb")as f2:
            for i in tqdm(range(8580)):
                name = pickle.load(f1)
                dic["file_name"].append(name)
                data = pickle.load(f2)
                input = data
                input_shapes = []
                input_shapes.append(input.shape)

                #net.inputs[0].data[:] = input.astype(np.float32)
                net.inputs[0].data[:] = input
                 #   else:
                 #       raise ValueError(f"unknown type: form {input.dtype} to {i.data.dtype}")
                #size = os.path.getsize(model_file)
                #t02 = time.time()
                #pack_bmodel_context = (pack_bmodel_context_generator(model_file, net))
                #next(pack_bmodel_context)
                #print("pack_bmodel_context:", time.time()-t02)

                # if size > 0x10000000:
                #     print("Warning: {} is too large and will cost a long time. Please run in board".format(
                #         model_file))
                #     return {}

                #t1 = time.time()
                #preprocess_time = t1 - begin_time
                #print("preprocess_time:", preprocess_time)

                #t01 = time.time()
                dyn_output_shapes = net.forward_dynamic(input_shapes)
                #print("net.forward_dynamic:",time.time() - t01)
                #t03 = time.time()


                outputs[net.outputs[0].name] = np.array(net.outputs[0].data)
                #try:
                #next(pack_bmodel_context)
                #except StopIteration:
                #    pass

                #t2 = time.time()
                #calculate_time = t2 - t1
                #print('calculate_time:', calculate_time)

                # return outputs
                idx = postprocess(outputs)
                cls = id2label[idx]
                dic["label"].append(cls)
                #t3 = time.time()
                #postprocess_time = t3 - t2
                #print("postprocess_time:",postprocess_time)
            df = pd.DataFrame(dic)
            df.to_csv("data1.csv" , index = False)


if __name__ == '__main__':
    main()
