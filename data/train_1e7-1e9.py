#!/usr/bin/env python
import argparse, os
import numpy as np
from utils import *

#from glob import glob

import tensorflow as tf
from model import denoiser

parser = argparse.ArgumentParser(description='')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

args = parser.parse_args()

def denoiser_train(denoiser, lr):
    #
    # load noisy and clean data
    #
    print("[*] Loading data...")

    #--------------------------------------------------------------------------
    # plan B : apply log(x+1) to the raw value 
    #--------------------------------------------------------------------------
    noisy_data = np.load('./patches/osa_img_noisy_pats_1e+07.npy')
    clean_data = np.load('./patches/osa_img_clean_pats_1e+07.npy')

    print noisy_data.shape , clean_data.shape

    noisy_data = np.log(noisy_data + 1.)
    clean_data = np.log(clean_data + 1.)

    #
    # Train 
    #
    denoiser.train(noisy_data, clean_data, batch_size=args.batch_size, 
            ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr=lr)


def main(_):
    # checkpoint dir
    if not os.path.exists(args.ckpt_dir): os.makedirs(args.ckpt_dir)

    lr = args.lr * np.ones([args.epoch])
    small_lr_pos = int(args.epoch * 0.6) # reduce learning rate after 60% total epoch
    lr[small_lr_pos:] = lr[0] * 0.1 


    if args.use_gpu:
        # added to control the gpu memory
        print("Run Tensorflow [GPU]\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess) # init a denoiser class
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                #denoiser_test(model)
                pass
            else:
                print('[!]Unknown phase')
                sys.exit(1)
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
