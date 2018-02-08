#!/usr/bin/env python
import numpy as np
import scipy.io as spio


def cal_psnr(mat1, mat2):
    #
    # assert the data range is 0 - 50 after log10()
    #
    mse = ((mat1.astype(np.float) - mat2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(50 ** 2 / mse)
    return psnr


def print_psnr(noisy_data, clean_data):
    noisymat = spio.loadmat(noisy_data, squeeze_me=True) # the output is a dict
    noisyData = noisymat['currentImage'] 

    cleanmat = spio.loadmat(clean_data, squeeze_me=True) # the output is a dict
    cleanData = cleanmat['currentImage'] 

    print('RawMax\nNoisy\t\tClean')
    print('{}\t{}'.format(noisyData.max(), cleanData.max()))

    noisyData = np.log(noisyData + 1.)
    cleanData = np.log(cleanData + 1.)

    print('\nlog(x+1)\nNoisy\t\tClean')
    print('{}\t{}'.format(noisyData.max(), cleanData.max()))

    print('\nPeak SNR (vs 1e9) : {}'.format(cal_psnr(noisyData, cleanData)))



def main():
    #-------------------
    # 1e8 : image 1 
    #-------------------
    print('\n----------\nimage 1 for 1e8 simulation\n----------\n')
    noisy_data = './osa_data/1e+08/1/osa_phn1e+08_test1_img1.mat'
    clean_data = './osa_data/1e+09/osa_1e9_img1.mat'

    print_psnr(noisy_data, clean_data)

    #-------------------
    # 1e8 : image 50 
    #-------------------
    print('\n----------\nimage 50 for 1e8 simulation\n----------\n')
    noisy_data = './osa_data/1e+08/1/osa_phn1e+08_test1_img50.mat'
    clean_data = './osa_data/1e+09/osa_1e9_img50.mat'

    print_psnr(noisy_data, clean_data)

    #-------------------
    # 1e7 : image 1 
    #-------------------
    print('\n----------\nimage 1 for 1e7 simulation\n----------\n')
    noisy_data = './osa_data/1e+07/1/osa_phn1e+07_test1_img1.mat'
    clean_data = './osa_data/1e+09/osa_1e9_img1.mat'

    print_psnr(noisy_data, clean_data)

    #-------------------
    # 1e7 : image 50 
    #-------------------
    print('\n----------\nimage 50 for 1e7 simulation\n----------\n')
    noisy_data = './osa_data/1e+07/1/osa_phn1e+07_test1_img50.mat'
    clean_data = './osa_data/1e+09/osa_1e9_img50.mat'

    print_psnr(noisy_data, clean_data)

    #-------------------
    # 1e6 : image 1 
    #-------------------
    print('\n----------\nimage 1 for 1e6 simulation\n----------\n')
    noisy_data = './osa_data/1e+06/1/osa_phn1e+06_test1_img1.mat'
    clean_data = './osa_data/1e+09/osa_1e9_img1.mat'

    print_psnr(noisy_data, clean_data)

    #-------------------
    # 1e6 : image 50 
    #-------------------
    print('\n----------\nimage 50 for 1e6 simulation\n----------\n')
    noisy_data = './osa_data/1e+06/1/osa_phn1e+06_test1_img50.mat'
    clean_data = './osa_data/1e+09/osa_1e9_img50.mat'

    print_psnr(noisy_data, clean_data)

    #-------------------
    # 1e5 : image 1 
    #-------------------
    print('\n----------\nimage 1 for 1e5 simulation\n----------\n')
    noisy_data = './osa_data/1e+05/1/osa_phn1e+05_test1_img1.mat'
    clean_data = './osa_data/1e+09/osa_1e9_img1.mat'

    print_psnr(noisy_data, clean_data)

    #-------------------
    # 1e5 : image 50 
    #-------------------
    print('\n----------\nimage 50 for 1e5 simulation\n----------\n')
    noisy_data = './osa_data/1e+05/1/osa_phn1e+05_test1_img50.mat'
    clean_data = './osa_data/1e+09/osa_1e9_img50.mat'

    print_psnr(noisy_data, clean_data)


if __name__ == '__main__':
    main()
