#!/usr/bin/env python
import argparse, os
import numpy as np
import tensorflow as tf
import scipy.io as spio

from model import denoiser
from utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')

args = parser.parse_args()

#parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
#parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')








def denoiser_test(denoiser):
    ##test_files = glob('./data/test/{}/*.png'.format(args.test_set))
    ##denoiser.test(test_files, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)

    #noisy_data = np.load('./patches/osa_img_noisy_pats_1e+05_quicktest.npy')
    #clean_data = np.load('./patches/osa_img_clean_pats_1e+05_quicktest.npy')
    #print noisy_data.shape , clean_data.shape


    noisy_data = './osa_data/1e+05/1/osa_phn1e+05_test1_img1.mat'
    clean_data = './osa_data/1e+09/osa_1e9_img1.mat'

    noisymat = spio.loadmat(noisy_data, squeeze_me=True) # the output is a dict
    cleanmat = spio.loadmat(clean_data, squeeze_me=True) # the output is a dict

    noisyData = noisymat['currentImage'] 
    cleanData = cleanmat['currentImage'] 

    (im_h, im_w) = noisyData.shape
    print noisyData.shape

                noisyData = np.reshape(noisyData, (im_h, im_w, 1))  # extend one dimension
                cleanData = np.reshape(cleanData, (im_h, im_w, 1))  # extend one dimension
    #test_image
    #denoiser.test(test_files, ckpt_dir=args.ckpt_dir, save_dir=args.test_dir)



def main(_):
    if args.use_gpu:
        # added to control the gpu memory
        print("Run Tensorflow [GPU]\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess)
            denoiser_test(model)
    else:
        ##print("CPU\n")
        print "CPU Not supported yet!"
        sys.exit(1)
        ##with tf.Session() as sess:
        ##    model = denoiser(sess, sigma=args.sigma)
        ##    if args.phase == 'train':
        ##        denoiser_train(model, lr=lr)
        ##    elif args.phase == 'test':
        ##        denoiser_test(model)
        ##    else:
        ##        print('[!]Unknown phase')
        ##        exit(0)


if __name__ == '__main__':
    tf.app.run()
