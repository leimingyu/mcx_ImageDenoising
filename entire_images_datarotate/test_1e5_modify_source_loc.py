#!/usr/bin/env python
import argparse
import os
import numpy as np
import tensorflow as tf
import scipy.io as spio

from model import denoiser

parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '--checkpoint_dir',
    dest='ckpt_dir',
    default='./checkpoint',
    help='models are saved here')
parser.add_argument(
    '--batch_size',
    dest='batch_size',
    type=int,
    default=64,
    help='# images in batch')
parser.add_argument(
    '--use_gpu',
    dest='use_gpu',
    type=int,
    default=1,
    help='gpu flag, 1 for GPU and 0 for CPU')

args = parser.parse_args()


def denoiser_test(denoiser):

    #--------------------------------------------------------------------------
    # plan B : apply log(x+1) to the raw value
    #--------------------------------------------------------------------------

    #------------
    # 1e8
    #------------
    noisy1  = '../entire_images/modify_source_loc/osa_data/1e+08/1/osa_phn1e+08_test1_img1.mat'
    noisy50 = '../entire_images/modify_source_loc/osa_data/1e+08/1/osa_phn1e+08_test1_img50.mat'

    noisymat1  = spio.loadmat(noisy1, squeeze_me=True)  # the output is a dict
    noisymat50 = spio.loadmat(noisy50, squeeze_me=True)  # the output is a dict

    noisyData1 = noisymat1['currentImage']
    noisyData50 = noisymat50['currentImage']

    (im_h, im_w) = noisyData1.shape

    noisyData1  = np.reshape(noisyData1, (im_h, im_w, 1))
    noisyData50 = np.reshape(noisyData50, (im_h, im_w, 1))

    # normalize data
    noisyData1  = np.log(noisyData1 + 1.)
    noisyData50 = np.log(noisyData50 + 1.)

    input_noisy1 = np.zeros((1, im_h, im_w, 1), dtype=np.float32)  # 4D matrix
    input_noisy50 = np.zeros((1, im_h, im_w, 1), dtype=np.float32)  # 4D matrix

    # update
    input_noisy1[0, :, :, :] = noisyData1
    input_noisy50[0, :, :, :] = noisyData50

    denoiser.test(input_noisy1, ckpt_dir=args.ckpt_dir,
                  outFile='1e5model-1e8-log_test1_img1.mat')
    denoiser.test(input_noisy50, ckpt_dir=args.ckpt_dir,
                  outFile='1e5model-1e8-log_test1_img50.mat')

    #------------
    # 1e7
    #------------
    noisy1  = '../entire_images/modify_source_loc/osa_data/1e+07/1/osa_phn1e+07_test1_img1.mat'
    noisy50 = '../entire_images/modify_source_loc/osa_data/1e+07/1/osa_phn1e+07_test1_img50.mat'

    noisymat1 = spio.loadmat(noisy1, squeeze_me=True)  # the output is a dict
    noisymat50 = spio.loadmat(noisy50, squeeze_me=True)  # the output is a dict

    noisyData1 = noisymat1['currentImage']
    noisyData50 = noisymat50['currentImage']

    (im_h, im_w) = noisyData1.shape

    noisyData1 = np.reshape(noisyData1, (im_h, im_w, 1)
                            )  # extend one dimension
    noisyData50 = np.reshape(
        noisyData50, (im_h, im_w, 1))  # extend one dimension

    # normalize data
    noisyData1 = np.log(noisyData1 + 1.)
    noisyData50 = np.log(noisyData50 + 1.)

    input_noisy1 = np.zeros((1, im_h, im_w, 1), dtype=np.float32)  # 4D matrix
    input_noisy50 = np.zeros((1, im_h, im_w, 1), dtype=np.float32)  # 4D matrix

    # update
    input_noisy1[0, :, :, :] = noisyData1
    input_noisy50[0, :, :, :] = noisyData50

    denoiser.test(input_noisy1, ckpt_dir=args.ckpt_dir,
                  outFile='1e5model-1e7-log_test1_img1.mat')
    denoiser.test(input_noisy50, ckpt_dir=args.ckpt_dir,
                  outFile='1e5model-1e7-log_test1_img50.mat')

    #------------
    # 1e6
    #------------
    noisy1  = '../entire_images/modify_source_loc/osa_data/1e+06/1/osa_phn1e+06_test1_img1.mat'
    noisy50 = '../entire_images/modify_source_loc/osa_data/1e+06/1/osa_phn1e+06_test1_img50.mat'

    noisymat1 = spio.loadmat(noisy1, squeeze_me=True)  # the output is a dict
    noisymat50 = spio.loadmat(noisy50, squeeze_me=True)  # the output is a dict

    noisyData1 = noisymat1['currentImage']
    noisyData50 = noisymat50['currentImage']

    (im_h, im_w) = noisyData1.shape

    noisyData1 = np.reshape(noisyData1, (im_h, im_w, 1)
                            )  # extend one dimension
    noisyData50 = np.reshape(
        noisyData50, (im_h, im_w, 1))  # extend one dimension

    # normalize data
    noisyData1 = np.log(noisyData1 + 1.)
    noisyData50 = np.log(noisyData50 + 1.)

    input_noisy1 = np.zeros((1, im_h, im_w, 1), dtype=np.float32)  # 4D matrix
    input_noisy50 = np.zeros((1, im_h, im_w, 1), dtype=np.float32)  # 4D matrix

    # update
    input_noisy1[0, :, :, :] = noisyData1
    input_noisy50[0, :, :, :] = noisyData50

    denoiser.test(input_noisy1, ckpt_dir=args.ckpt_dir,
                  outFile='1e5model-1e6-log_test1_img1.mat')
    denoiser.test(input_noisy50, ckpt_dir=args.ckpt_dir,
                  outFile='1e5model-1e6-log_test1_img50.mat')

    #------------
    # 1e5
    #------------
    noisy1  = '../entire_images/modify_source_loc/osa_data/1e+05/1/osa_phn1e+05_test1_img1.mat'
    noisy50 = '../entire_images/modify_source_loc/osa_data/1e+05/1/osa_phn1e+05_test1_img50.mat'

    noisymat1 = spio.loadmat(noisy1, squeeze_me=True)  # the output is a dict
    noisymat50 = spio.loadmat(noisy50, squeeze_me=True)  # the output is a dict

    noisyData1 = noisymat1['currentImage']
    noisyData50 = noisymat50['currentImage']

    (im_h, im_w) = noisyData1.shape

    noisyData1 = np.reshape(noisyData1, (im_h, im_w, 1)
                            )  # extend one dimension
    noisyData50 = np.reshape(
        noisyData50, (im_h, im_w, 1))  # extend one dimension

    # normalize data
    noisyData1 = np.log(noisyData1 + 1.)
    noisyData50 = np.log(noisyData50 + 1.)

    input_noisy1 = np.zeros((1, im_h, im_w, 1), dtype=np.float32)  # 4D matrix
    input_noisy50 = np.zeros((1, im_h, im_w, 1), dtype=np.float32)  # 4D matrix

    # update
    input_noisy1[0, :, :, :] = noisyData1
    input_noisy50[0, :, :, :] = noisyData50

    denoiser.test(input_noisy1, ckpt_dir=args.ckpt_dir,
                  outFile='1e5model-1e5-log_test1_img1.mat')
    denoiser.test(input_noisy50, ckpt_dir=args.ckpt_dir,
                  outFile='1e5model-1e5-log_test1_img50.mat')


def main(_):
    if args.use_gpu:
        # added to control the gpu memory
        print("Run Tensorflow [GPU]\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess)
            denoiser_test(model)
    else:
        print "CPU Not supported yet!"
        sys.exit(1)


if __name__ == '__main__':
    tf.app.run()
