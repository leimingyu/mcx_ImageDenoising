#!/usr/bin/env python

import glob
import argparse
import sys
import os

# load mat in python
# http://www.blogforbrains.com/blog/2014/9/6/loading-matlab-mat-data-in-python
import scipy.io as spio
import numpy as np

#import random
#from utils import *

#------------------------------------------------------------------------------
# For MCX simulation results, the value could range from 0 to a quite large
# number in floating point
# (1) For each image file, i.e., xxx_image??.mat, the ?? number stands for
# the horizontal slice in 3D. We need to find their corresponding ground truth in 1e+9!
#------------------------------------------------------------------------------


parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '--batch_size',
    dest='bat_size',
    type=int,
    default=64,
    help='batch size')
parser.add_argument(
    '--save_dir',
    dest='save_dir',
    default='./patches',
    help='dir of patches')
args = parser.parse_args()


def gen_osa_data(photon_vol,
                 test_num=100,
                 data_dir='../data/osa_data',
                 ground_truth='../data/osa_data/1e+09',
                 ground_truth_img_prefix='osa_1e9_img'):
    '''
    Each simulation results in 100 different 3D voxel image
    The ground_truth image name example:osa_1e9_img1.mat and osa_1e9_img100.mat
    For osa data sets, we used 100x100x100 3D voxel. The data size is determined.
    '''
    target_dir = data_dir + '/' + photon_vol  # locate the simulation folder

    #--------------------------------------------------------------------------
    # 1) count the number of patches
    #--------------------------------------------------------------------------
    img_count = 0
    for test_id in xrange(
            1, test_num + 1):  # there are 100 tests for each photon simulation
        files_in_dir = target_dir + '/' + \
            str(test_id) + '/*.mat'   # load mat files
        filepaths = glob.glob(files_in_dir)    # form the file path
        img_count = img_count + len(filepaths)

    print "[LOG] There are %d images." % img_count

    if img_count % args.bat_size != 0:  # if can't be evenly dived by batch size
        numPatches = (img_count / args.bat_size + 1) * args.bat_size
    else:
        numPatches = img_count

    print "[LOG] total patches = %d , batch size = %d, total batches = %d" % \
          (numPatches, args.bat_size, numPatches / args.bat_size)

    # data matrix 4-D
    inputs_noisy = np.zeros((numPatches, 100, 100, 1),
                            dtype=np.float32)  # osa image is 100x100
    inputs_clean = np.zeros((numPatches, 100, 100, 1), dtype=np.float32)

    #--------------------------------------------------------------------------
    # 2) generate the patches
    #--------------------------------------------------------------------------
    patch_count = 0

    for test_id in xrange(1, test_num + 1):
        files_in_dir = target_dir + '/' + \
            str(test_id) + '/*.mat'   # load mat files
        filepaths = glob.glob(files_in_dir)    # form the file path

        img_prefix = target_dir + '/' + str(test_id) + '/osa_phn' + photon_vol + \
            '_test' + str(test_id) + '_img'

        for _, noisyfile in enumerate(filepaths):
            noisymat = spio.loadmat(noisyfile, squeeze_me=True)
            noisyData = noisymat['currentImage']

            img_prefix_len = len(img_prefix)
            # remove the prefix, then the suffix ".mat"
            img_id = int(noisyfile[img_prefix_len:][:-4])
            cleanfile = ground_truth + '/' + \
                ground_truth_img_prefix + str(img_id) + '.mat'
            cleanmat = spio.loadmat(cleanfile, squeeze_me=True)
            cleanData = cleanmat['currentImage']

            if noisyData.shape[0] != cleanData.shape[0] or noisyData.shape[1] != cleanData.shape[1]:
                print('Error! Noisy data size is different from clean data size!')
                sys.exit(1)

            (im_h, im_w) = noisyData.shape

            noisyData = np.reshape(
                noisyData, (im_h, im_w, 1))  # extend one dimension
            cleanData = np.reshape(
                cleanData, (im_h, im_w, 1))  # extend one dimension
            # print noisyData.shape

            inputs_noisy[patch_count, :, :, :] = noisyData[:, :, :]
            inputs_clean[patch_count, :, :, :] = cleanData[:, :, :]

            patch_count = patch_count + 1

    print '[LOG] %d patches are generated!' % patch_count

    #--------------------------------------------------------------------------
    # 3) pad the batch
    #--------------------------------------------------------------------------
    if patch_count < numPatches:
        print '[LOG] padding the batch ... '
        to_pad = numPatches - patch_count
        inputs_noisy[-to_pad:, :, :, :] = inputs_noisy[:to_pad, :, :, :]
        inputs_clean[-to_pad:, :, :, :] = inputs_clean[:to_pad, :, :, :]

    #--------------------------------------------------------------------------
    # 4) output
    #--------------------------------------------------------------------------
    print '[LOG] saving data to disk ... '
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    np.save(
        os.path.join(
            args.save_dir,
            "osa_img_noisy_pats_" +
            photon_vol),
        inputs_noisy)
    np.save(
        os.path.join(
            args.save_dir,
            "osa_img_clean_pats_" +
            photon_vol),
        inputs_clean)

    print '[LOG] Done! '
    print '[LOG] Check %s for the output.' % args.save_dir
    print "[LOG] size of inputs tensor = " + str(inputs_noisy.shape)


if __name__ == '__main__':

    print '\nGenerating osa data [1e+05]'
    gen_osa_data('1e+05')

    # print '\nGenerating osa data [1e+06]'
    # gen_osa_data('1e+06')

    # print '\nGenerating osa data [1e+07]'
    # gen_osa_data('1e+07')

    # print '\nGenerating osa data [1e+08]'
    # gen_osa_data('1e+08')
