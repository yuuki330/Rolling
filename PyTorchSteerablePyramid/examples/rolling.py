# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-12-04

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append('../')
import skimage.io

import argparse
import time
import numpy as np
import torch
import torchvision
import cv2

from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
import steerable.utils as utils

from scipy.io import wavfile

video_fps = 0
video_time = 0

def load_video_batch(video_file, batch_size):
    ToPIL = torchvision.transforms.ToPILImage()
    Grayscale = torchvision.transforms.Grayscale()
    # RandomCrop = torchvision.transforms.RandomCrop
    CenterCrop = torchvision.transforms.CenterCrop

    if not os.path.isfile(video_file):
        raise FileNotFoundError('Video file not found on disk: {}'.format(video_file))
    
    cap = cv2.VideoCapture(video_file)
    global video_fps
    global video_time
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_time = cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps

    print(f'video_width : {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}')
    print(f'video_height : {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}')
    print(f'video_fps : {cap.get(cv2.CAP_PROP_FPS)}')
    print(f'video_frame_count : {cap.get(cv2.CAP_PROP_FRAME_COUNT)}')
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    image_size = min(width, height)

    j = 0
    # im_all = np.zeros((num_frames, height, width), np.float32)
    im_batch = np.zeros((num_frames, batch_size, 1, image_size, image_size), np.float32)

    while True:
        ret,frame = cap.read()
        if ret == False:
            break

        im = ToPIL(frame)
        im = Grayscale(im)
        
        # for i in range(batch_size):
        #     im_batch[j][i][0] = RandomCrop(image_size)(im)
        im_batch[j,0,0] = CenterCrop(image_size)(im)
        im_batch[j] = im_batch[j,:,:,:]/225.

        j += 1
        if j == 100:
            break

    return im_batch


def make_grid_coeff(coeff, normalize=True, real=True):
    '''
    Visualization function for building a large image that contains the
    low-pass, high-pass and all intermediate levels in the steerable pyramid. 
    For the complex intermediate bands, the real part is visualized.
    
    Args:
        coeff (list): complex pyramid stored as list containing all levels
        normalize (bool, optional): Defaults to True. Whether to normalize each band
    
    Returns:
        np.ndarray: large image that contains grid of all bands and orientations
    '''
    M, N = coeff[1][0].shape
    Norients = len(coeff[1])
    out = np.zeros((M * 2 - coeff[-1].shape[0], Norients * N))
    currentx, currenty = 0, 0

    if real:
        for i in range(1, len(coeff[:-1])):
            for j in range(len(coeff[1])):
                tmp = coeff[i][j].real
                m, n = tmp.shape
                if normalize:
                    tmp = 255 * tmp/tmp.max()
                tmp[m-1,:] = 255
                tmp[:,n-1] = 255
                out[currentx:currentx+m,currenty:currenty+n] = tmp
                currenty += n
            currentx += coeff[i][0].shape[0]
            currenty = 0
    else:
        for i in range(1, len(coeff[:-1])):
            for j in range(len(coeff[1])):
                tmp = coeff[i][j].imag
                m, n = tmp.shape
                if normalize:
                    tmp = 255 * tmp/tmp.max()
                tmp[m-1,:] = 255
                tmp[:,n-1] = 255
                out[currentx:currentx+m,currenty:currenty+n] = tmp
                currenty += n
            currentx += coeff[i][0].shape[0]
            currenty = 0

    m, n = coeff[-1].shape
    out[currentx: currentx+m, currenty: currenty+n] = 255 * coeff[-1]/coeff[-1].max()
    out[0,:] = 255
    out[:,0] = 255
    return out.astype(np.uint8)

def scale_array(arr):
    max_val = np.max(arr)
    min_val = np.min(arr)
    scaled_arr = (2 * (arr - min_val) / (max_val - min_val)) - 1
    return scaled_arr

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, default='../assets/KitKat-60Hz-RollingShutter-Mary_MIDI-input.avi')
    parser.add_argument('--batch_size', type=int, default='1')
    # parser.add_argument('--image_size', type=int, default='720')
    parser.add_argument('--pyr_nlevels', type=int, default='5')
    parser.add_argument('--pyr_nbands', type=int, default='4')
    parser.add_argument('--pyr_scale_factor', type=int, default='2')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--visualize', type=bool, default=True)
    config = parser.parse_args()

    device = utils.get_device(config.device)


    pyr = SCFpyr_PyTorch(
        height=config.pyr_nlevels, 
        nbands=config.pyr_nbands,
        scale_factor=config.pyr_scale_factor, 
        device=device
    )

    ## grid表示
    # im_batch_numpy = load_video_batch(config.video_file, config.batch_size)
    # im_batch_torch = torch.from_numpy(im_batch_numpy[0]).to(device)
    # coeff = pyr.build(im_batch_torch)
    # coeff = utils.extract_from_batch(coeff, 0)
    
    # coeff_grid_real = make_grid_coeff(coeff, normalize=True)
    # coeff_grid_imag = make_grid_coeff(coeff, normalize=True, real=False)
    # cv2.imwrite('./output/test/image.jpg', (im_batch_numpy[0,0,0,]*255.).astype(np.uint8))
    # cv2.imwrite('./output/test/coeff_real.jpg', coeff_grid_real)
    # cv2.imwrite('./output/test/coeff_imag.jpg', coeff_grid_imag)


    ## signal計算
    im_batch_numpy = load_video_batch(config.video_file, config.batch_size)
    height, width = im_batch_numpy[0,0,0].shape
    im_size = min(height, width)
    im_batch_torch = torch.from_numpy(im_batch_numpy[0]).to(device)
    coeff_0 = pyr.build(im_batch_torch)
    coeff_0 = utils.extract_from_batch(coeff_0, 0)

    sampling_rate = 2200
    phi_ = np.zeros((int(61920*video_time)))
    print(f'phi_ shape: {phi_.shape}')

    H = 0 

    # for t in range(1, im_batch_numpy.shape[0]):
    for t in range(1, 50):
        im_batch_torch = torch.from_numpy(im_batch_numpy[t]).to(device)
        coeff = pyr.build(im_batch_torch)
        coeff = utils.extract_from_batch(coeff, 0)
        phi_v = coeff
        band_signal = [[] for _ in range(len(coeff[1]))]

        for pyr_level in range(len(coeff)-2, 0, -1):
            # for nband in range(len(coeff[pyr_level])):
            for nband in range(1, 0, -1):
                phi = np.zeros((len(coeff[pyr_level][nband])))
                # if pyr_level == len(coeff)-2:
                #     H_t = len(phi)
                # else:
                #     H += len(phi)
                H += len(phi)
                for i in range(len(coeff[pyr_level][nband])):
                    for j in range(len(coeff[pyr_level][nband][i])):
                        phi_v[pyr_level][nband][i][j] =  coeff[pyr_level][nband][i][j].imag - coeff_0[pyr_level][nband][i][j].imag
                        phi_v[pyr_level][nband][i][j] = (coeff[pyr_level][nband][i][j].real) * (coeff[pyr_level][nband][i][j].real) * phi_v[pyr_level][nband][i][j]
                    phi[i] = np.sum(phi_v[pyr_level][nband][i]).real

                    # アップサンプリング倍率
                    upsampling_factor = im_size // len(phi)
                    # アップサンプリングを行う
                    phi = np.repeat(phi, upsampling_factor)

                    if i == im_size-1:
                        N_gap = sampling_rate // video_fps
                    else:
                        N_gap = 0
                    n = int(i + ((H + N_gap) * t))
                    print(f'H: {H}, N_gap: {N_gap}, t: {t}, i: {i}, n: {n}')
                    phi_[n] = phi[i]
                # print(phi_)
    #     if t == 0:
    #         tmp = np.array([0])

    #     t_signal = np.zeros(len(band_signal[0]))
    #     for nband in range(len(coeff[1])):
    #         t_signal += np.array(band_signal[nband]) 
    #     tmp = np.concatenate((tmp, t_signal))

    # signal = scale_array(tmp)
    # signal = signal-signal.mean()
    # signal = np.delete(signal, 0)
    # signal = np.round(signal * 32767).astype(np.int16)

    # print(signal)
    # print(len(signal))

    # # WAVファイルとして保存
    # output_file = "output.wav"
    # wavfile.write(output_file, sampling_rate, signal)