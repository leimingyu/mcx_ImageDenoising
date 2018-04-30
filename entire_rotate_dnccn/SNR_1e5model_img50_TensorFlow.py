#!/usr/bin/env python
import glob
import argparse, os
import numpy as np
import tensorflow as tf
import scipy.io as spio

from model import denoiser

parser = argparse.ArgumentParser(description='')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')

args = parser.parse_args()

#parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
#parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')


def Denoising(denoiser, X):
    (im_h, im_w) = X.shape

    X_new = np.reshape(X, (im_h, im_w, 1))  # extend one dimension

    # normalize data
    X_new = np.log(X_new + 1.)

    inputX = np.zeros((1, im_h, im_w, 1), dtype=np.float32) # 4D matrix

    # update
    inputX[0, :, :, :]  = X_new 

    return denoiser.test(inputX, ckpt_dir=args.ckpt_dir)

    



def denoiser_test(denoiser):
    print('Measure the SNR at the mid of voxel (image 50 on y axis for 100x100x100 volume).')

    #--------------------------------------------------------------------------
    # plan B : apply log(x+1) to the raw value 
    #--------------------------------------------------------------------------

    # 1) read all the img50 from ./osa_data_testing/1e+05/ 
    test_num = 100
    photon_vol = '1e+05'
    #target_dir = './osa_data_testing/1e+05' 
    target_dir = '../data/osa_data_testing/1e+05' 

    for test_id in xrange(1, test_num + 1): # there are 100 tests for each photon simulation
        #files_in_dir = target_dir + '/' + str(test_id) + '/*.mat'   # load mat files
        files_in_dir = target_dir + '/' + str(test_id) + '/*img50.mat'   # load mat files
        filepaths = glob.glob(files_in_dir)    # form the file path
        #print filepaths

        for i, noisyfile in enumerate(filepaths): # read each file
            noisymat = spio.loadmat(noisyfile, squeeze_me=True) 
            noisyData = noisymat['currentImage'] 
            (im_h, im_w) = noisyData.shape
            break

        break

    print "[LOG] test number = %d , im_h = %d, im_w = %d" % \
          (test_num, im_h, im_w)

    # data matrix 3-D
    denoiser_output = np.zeros((test_num, im_h, im_w), dtype=np.float32)
    
    for test_id in xrange(1, test_num + 1): # there are 100 tests for each photon simulation
        files_in_dir = target_dir + '/' + str(test_id) + '/*img50.mat'   # load mat files
        filepaths = glob.glob(files_in_dir)    # form the file path

        for i, noisyfile in enumerate(filepaths): # read each file
            noisymat = spio.loadmat(noisyfile, squeeze_me=True) 
            noisyData = noisymat['currentImage'] # 2d matrix : raw input

            noisyData = np.log(noisyData + 1.) # pre-process the input

            # Note: not an efficient way for testing
            model_output = Denoising(denoiser, noisyData) # tensorflow output

            model_output = np.exp(model_output) - 1. # we need to convert back from log(x + 1)

            denoiser_output[test_id-1, : , :] = model_output


    #
    # read denoiser_output: the 50th row of each image
    #
    target_image_row = np.zeros((test_num, im_w), dtype=np.float32)

    print "\nno coverting neg to 1e-8"

    for test_id in xrange(test_num):
        target_image_row[test_id, :] = denoiser_output[test_id, 49, :]

    #print target_image_row


    #
    # SNR : 20 x log10( mean / std) 
    #

    means = np.mean(target_image_row, axis=0) # along column
    stds  = np.std(target_image_row, axis=0)
    snr_results = 20 * np.log10(means / stds)

    for x in np.nditer(snr_results):
        print x

    
    print "\ncoverting neg to 1e-8"
    target_image_row[target_image_row < 0.0] = 1e-8

    means = np.mean(target_image_row, axis=0) # along column
    stds  = np.std(target_image_row, axis=0)
    snr_results = 20 * np.log10(means / stds)

    for x in np.nditer(snr_results):
        print x



    print "[LOG] Done!"





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
