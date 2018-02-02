#!/usr/bin/env python

import glob
import argparse
import sys, os
import numpy as np
# load mat in python
# http://www.blogforbrains.com/blog/2014/9/6/loading-matlab-mat-data-in-python
import scipy.io as spio
#from PIL import Image
#import PIL

#import random
#from utils import *

#------------------------------------------------------------------------------
# For MCX simulation results, the value could range from 0 to a quite large 
# number in floating point
# (1) For each image file, i.e., xxx_image??.mat, the ?? number stands for 
# the horizontal slice in 3D. We need to find their corresponding ground truth in 1e+9!
#  
#------------------------------------------------------------------------------


parser = argparse.ArgumentParser(description='')
parser.add_argument('--step', dest='step', type=int, default=0, help='step')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=40, help='patch size')
parser.add_argument('--stride', dest='stride', type=int, default=10, help='stride')
parser.add_argument('--batch_size', dest='bat_size', type=int, default=128, help='batch size')
parser.add_argument('--save_dir', dest='save_dir', default='./patches', help='dir of patches')
args = parser.parse_args()

###parser.add_argument('--src_dir', dest='src_dir', default='./data/Train400', help='dir of data')
##parser.add_argument('--save_dir', dest='save_dir', default='./data', help='dir of patches')
##parser.add_argument('--patch_size', dest='pat_size', type=int, default=40, help='patch size')   # leiming: patch size can be considered as the filter size
##parser.add_argument('--stride', dest='stride', type=int, default=10, help='stride')
##parser.add_argument('--batch_size', dest='bat_size', type=int, default=128, help='batch size')
### check output arguments
##parser.add_argument('--from_file', dest='from_file', default="./data/img_clean_pats.npy", help='get pic from file')
##parser.add_argument('--num_pic', dest='num_pic', type=int, default=10, help='number of pic to pick')

def generate_patches():
    data_dir = './osa_data'

    ground_truth = './osa_data/1e+09'
    ground_truth_img_prefix = 'osa_1e9_img'  # e.g., osa_1e9_img1.mat and osa_1e9_img100.mat

    #photon_vol      = '1e+05'
    photon_vol_list = ['1e+05', '1e+06', '1e+07', '1e+08']

    test_num = 100 # each simulation results in 100 different 3D voxel image

    patch_count = 0

    #--------------------------------------------------------------------------
    # 1) count the number of patches
    #--------------------------------------------------------------------------
    for photon_vol in photon_vol_list:
        target_dir = data_dir + '/' + photon_vol  # locate the simulation folder

        for test_id in xrange(1, test_num + 1): # there are 100 tests for each photon simulation
            files_in_dir = target_dir + '/' + str(test_id) + '/*.mat'   # load mat files
            filepaths = glob.glob(files_in_dir)    # form the file path
            #print filepaths[0]

            img_prefix = target_dir + '/' + str(test_id) + '/osa_phn' + photon_vol + \
                    '_test' + str(test_id) + '_img'

            for i, noisyfile in enumerate(filepaths): # read each file
                #noisymat = spio.loadmat(noisyfile, squeeze_me=True) # the output is a dict
                #noisyData = noisymat['currentImage'] 
                #(im_h, im_w) = noisyData.shape
                (im_h, im_w) = (100, 100)  # Note: quick hack to save time

                for x in range(0 + args.step, (im_h - args.pat_size), args.stride):
                    for y in range(0 + args.step, (im_w - args.pat_size), args.stride):
                        patch_count = patch_count + 1


    if patch_count % args.bat_size != 0: # if can't be evenly dived by batch size
        numPatches = (patch_count / args.bat_size + 1) * args.bat_size
    else:
        numPatches = patch_count 

    print "[LOG] total patches = %d , batch size = %d, total batches = %d" % \
          (numPatches, args.bat_size, numPatches / args.bat_size)


    # data matrix 4-D
    inputs_noisy = np.zeros((numPatches, args.pat_size, args.pat_size, 1), dtype=np.float32)
    inputs_clean = np.zeros((numPatches, args.pat_size, args.pat_size, 1), dtype=np.float32)


    #--------------------------------------------------------------------------
    # 2) generate the patches
    #--------------------------------------------------------------------------
    patch_count = 0

    for photon_vol in photon_vol_list:
        target_dir = data_dir + '/' + photon_vol  # locate the simulation folder

        for test_id in xrange(1, test_num + 1):
            files_in_dir = target_dir + '/' + str(test_id) + '/*.mat'   # load mat files
            filepaths = glob.glob(files_in_dir)    # form the file path

            img_prefix = target_dir + '/' + str(test_id) + '/osa_phn' + photon_vol + \
                    '_test' + str(test_id) + '_img'

            for i, noisyfile in enumerate(filepaths):
                noisymat = spio.loadmat(noisyfile, squeeze_me=True) 
                noisyData = noisymat['currentImage'] 

                img_prefix_len = len(img_prefix)
                img_id = int(noisyfile[img_prefix_len:][:-4]) # remove the prefix, then the suffix ".mat"
                cleanfile = ground_truth + '/' + ground_truth_img_prefix + str(img_id) + '.mat' 
                cleanmat = spio.loadmat(cleanfile, squeeze_me=True)
                cleanData = cleanmat['currentImage']  

                if noisyData.shape[0] <> cleanData.shape[0] or noisyData.shape[1] <> cleanData.shape[1]:
                    print('Error! Noisy data size is different from clean data size!')
                    sys.exit(1)

                #print noisyData.shape, cleanData.shape
                (im_h, im_w) = noisyData.shape
                #print noisyData[0, :10]
                #print type(noisyData[0,0])

                noisyData = np.reshape(noisyData, (im_h, im_w, 1))  # extend one dimension
                cleanData = np.reshape(cleanData, (im_h, im_w, 1))  # extend one dimension
                #print noisyData.shape
                

                for x in range(0 + args.step, (im_h - args.pat_size), args.stride):
                    for y in range(0 + args.step, (im_w - args.pat_size), args.stride):
                        inputs_noisy[patch_count, :, :, :] = noisyData[x:x+args.pat_size, y:y+args.pat_size, :]
                        inputs_clean[patch_count, :, :, :] = cleanData[x:x+args.pat_size, y:y+args.pat_size, :]

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

    np.save(os.path.join(args.save_dir, "osa_img_noisy_pats"), inputs_noisy)
    np.save(os.path.join(args.save_dir, "osa_img_clean_pats"), inputs_clean)

    print '[LOG] Done! '
    print '[LOG] Check %s for the output.' % args.save_dir
    print "[LOG] size of inputs tensor = " + str(inputs_noisy.shape)


def gen_osa_data(photon_vol):
    data_dir = './osa_data'

    ground_truth = './osa_data/1e+09'
    ground_truth_img_prefix = 'osa_1e9_img'  # e.g., osa_1e9_img1.mat and osa_1e9_img100.mat

    test_num = 100 # each simulation results in 100 different 3D voxel image

    patch_count = 0

    target_dir = data_dir + '/' + photon_vol  # locate the simulation folder

    #--------------------------------------------------------------------------
    # 1) count the number of patches
    #--------------------------------------------------------------------------

    for test_id in xrange(1, test_num + 1): # there are 100 tests for each photon simulation
        files_in_dir = target_dir + '/' + str(test_id) + '/*.mat'   # load mat files
        filepaths = glob.glob(files_in_dir)    # form the file path
        #print filepaths[0]

        img_prefix = target_dir + '/' + str(test_id) + '/osa_phn' + photon_vol + \
                '_test' + str(test_id) + '_img'

        for i, noisyfile in enumerate(filepaths): # read each file
            #noisymat = spio.loadmat(noisyfile, squeeze_me=True) # the output is a dict
            #noisyData = noisymat['currentImage'] 
            #(im_h, im_w) = noisyData.shape
            (im_h, im_w) = (100, 100)  # Note: quick hack to save time

            for x in range(0 + args.step, (im_h - args.pat_size), args.stride):
                for y in range(0 + args.step, (im_w - args.pat_size), args.stride):
                    patch_count = patch_count + 1


    if patch_count % args.bat_size != 0: # if can't be evenly dived by batch size
        numPatches = (patch_count / args.bat_size + 1) * args.bat_size
    else:
        numPatches = patch_count 

    print "[LOG] total patches = %d , batch size = %d, total batches = %d" % \
          (numPatches, args.bat_size, numPatches / args.bat_size)


    # data matrix 4-D
    inputs_noisy = np.zeros((numPatches, args.pat_size, args.pat_size, 1), dtype=np.float32)
    inputs_clean = np.zeros((numPatches, args.pat_size, args.pat_size, 1), dtype=np.float32)


    #--------------------------------------------------------------------------
    # 2) generate the patches
    #--------------------------------------------------------------------------
    patch_count = 0

    for test_id in xrange(1, test_num + 1):
        files_in_dir = target_dir + '/' + str(test_id) + '/*.mat'   # load mat files
        filepaths = glob.glob(files_in_dir)    # form the file path

        img_prefix = target_dir + '/' + str(test_id) + '/osa_phn' + photon_vol + \
                '_test' + str(test_id) + '_img'

        for i, noisyfile in enumerate(filepaths):
            noisymat = spio.loadmat(noisyfile, squeeze_me=True) 
            noisyData = noisymat['currentImage'] 

            img_prefix_len = len(img_prefix)
            img_id = int(noisyfile[img_prefix_len:][:-4]) # remove the prefix, then the suffix ".mat"
            cleanfile = ground_truth + '/' + ground_truth_img_prefix + str(img_id) + '.mat' 
            cleanmat = spio.loadmat(cleanfile, squeeze_me=True)
            cleanData = cleanmat['currentImage']  

            if noisyData.shape[0] <> cleanData.shape[0] or noisyData.shape[1] <> cleanData.shape[1]:
                print('Error! Noisy data size is different from clean data size!')
                sys.exit(1)

            #print noisyData.shape, cleanData.shape
            (im_h, im_w) = noisyData.shape
            #print noisyData[0, :10]
            #print type(noisyData[0,0])

            noisyData = np.reshape(noisyData, (im_h, im_w, 1))  # extend one dimension
            cleanData = np.reshape(cleanData, (im_h, im_w, 1))  # extend one dimension
            #print noisyData.shape
            

            for x in range(0 + args.step, (im_h - args.pat_size), args.stride):
                for y in range(0 + args.step, (im_w - args.pat_size), args.stride):
                    inputs_noisy[patch_count, :, :, :] = noisyData[x:x+args.pat_size, y:y+args.pat_size, :]
                    inputs_clean[patch_count, :, :, :] = cleanData[x:x+args.pat_size, y:y+args.pat_size, :]

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

    np.save(os.path.join(args.save_dir, "osa_img_noisy_pats_" + photon_vol), inputs_noisy)
    np.save(os.path.join(args.save_dir, "osa_img_clean_pats_" + photon_vol), inputs_clean)

    print '[LOG] Done! '
    print '[LOG] Check %s for the output.' % args.save_dir
    print "[LOG] size of inputs tensor = " + str(inputs_noisy.shape)


if __name__ == '__main__':
    #generate_patches()

    #photon_vol_list = ['1e+05', '1e+06', '1e+07', '1e+08']
    print '\nGenerating osa data [1e+05]'
    gen_osa_data('1e+05')

    print '\nGenerating osa data [1e+06]'
    gen_osa_data('1e+06')

    print '\nGenerating osa data [1e+07]'
    gen_osa_data('1e+07')

    print '\nGenerating osa data [1e+08]'
    gen_osa_data('1e+08')
