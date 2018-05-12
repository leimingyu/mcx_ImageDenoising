#!/usr/bin/env python
import glob
import os
import numpy as np
import scipy.io as spio

from utils import *


def main():
    print('Measure the SNR at the mid of voxel (image 50 on y axis for 100x100x100 volume).')

    #--------------------------------------------------------------------------
    # plan B : apply log(x+1) to the raw value 
    #--------------------------------------------------------------------------

    # 1) read all the img50 from ./osa_data_testing/1e+05/ 
    test_num = 10
    photon_vol = '1e+09'
    target_dir = './osa_data/1e+09' 

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

        # NOTE: one file per folder
        for i, noisyfile in enumerate(filepaths): # read each file
            noisymat = spio.loadmat(noisyfile, squeeze_me=True) 
            noisyData = noisymat['currentImage'] # 2d matrix : raw data
            denoiser_output[test_id-1, : , :] = noisyData  


    #
    # read denoiser_output: the 50th row of each image
    #
    target_image_row = np.zeros((test_num, im_w), dtype=np.float32)

    print "\n Working on the raw input ..."

    for test_id in xrange(test_num):
        target_image_row[test_id, :] = denoiser_output[test_id, 49, :]

    #print target_image_row


    #
    # SNR : 20 x log10( mean / std) 
    #

    means = np.mean(target_image_row, axis=0) # along column
    stds  = np.std(target_image_row, axis=0)
    snr_results = 20 * np.log10(means / stds)

    print "SNR for the mid of image 50 on the y-axis"
    for x in np.nditer(snr_results):
        print x


    print "[LOG] Done!"



if __name__ == '__main__':
    main() 
