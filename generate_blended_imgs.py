#!/usr/bin/env python3

import cv2
import os
from os import path as osp
import json
import argparse as ap
import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
from functools import reduce
from skimage.io import imread

def get_figure(img):
    plt.figure()
    plt.imshow(img)

class Blender(object):

    def __init__(self, img_content_path, img_style_path, params={}):
        self.img_content = imread(img_content_path)/255.
        self.img_style = imread(img_style_path)/255.
        content_shape = self.img_content.shape
        self.img_style = cv2.resize(self.img_style,(content_shape[1], content_shape[0]))
        print(self.img_content.shape, self.img_style.shape)
        self.params = params

    def poisson_blend(self):
        raise NotImplementedError('Poisson Blending not yet implemented!')

    def get_laplacian_pyramid(self, img, scale=4):
        gp_tmp = img.copy()
        gp = [gp_tmp]
        for i in range(scale):
            gp_tmp = cv2.pyrDown(gp_tmp)
            gp.append(gp_tmp)
        #Generate Laplacian pyramid
        lp = [gp[-1]]
        for i in range(len(gp)-1, 0, -1):
            g_tmp = cv2.pyrUp(gp[i])
            lp.append(gp[i-1] - g_tmp)
        return lp
    
    @staticmethod
    def  power_transform(img, rho=1.0):
        return np.sign(img)*np.power(np.abs(img), rho)
    
    def power_weighted_mean(self):
        w1 = self.params.get('weight', 1.0)
        w2 = 1-w1
        rho_range = self.params.get('rho', [0.5])
        scale = self.params.get('scale', 4)
        lp_content = self.get_laplacian_pyramid(self.img_content, scale)
        lp_style = self.get_laplacian_pyramid(self.img_style, scale)
        rho_blended = {}
        for rho in rho_range:    
            blended_lp = [w1 * Blender.power_transform(i, rho) + w2 * Blender.power_transform(j, rho) for i,j in zip(lp_content, lp_style)]
            blended_scales = [Blender.power_transform(i, 1./rho) for i in blended_lp]
            #reconstruct
            blended_img = blended_scales[0].copy()
            for i in range(1, len(blended_scales)):
                #print(i)
                blended_img = cv2.pyrUp(blended_img)
                # print(blended_img.shape, blended_scales[i].shape)
                blended_img += blended_scales[i]
            rho_blended[rho] = blended_img
        return rho_blended

    def blend(self, blending_type):
        if blending_type == 'poisson_blend':
            self.processed_img = self.poisson_blend()
        elif blending_type == 'pwm':
            self.processed_img = self.power_weighted_mean()
        else:
            raise NotImplementedError('Blending type : %s not implented' % (self.blending_type))
        return self.processed_img

def build_parser():
    parser = ap.ArgumentParser()
    parser.add_argument('-ic', dest='content_img', help='Content image directory')
    parser.add_argument('-is', dest='style_img', help='Style/Attribute Image directory')
    parser.add_argument('--param_file', help='Path to parameter file')
    parser.add_argument('--blend_type', help='Blend type: (poisson_blend, pwm : power_weighted_mean)', default='pwm')
    return parser 

if __name__ == '__main__':
    args = build_parser().parse_args()
    with open(args.param_file, 'r') as f:
        params = json.load(f)
    for r1,_,f1 in os.walk(args.content_img):
        for file1 in f1:
            for r2,_,f2 in os.walk(args.style_img):
                for file2 in f2:
                    print(file1, file2)
                    fname ='-'.join([osp.splitext(file1)[0], osp.splitext(file2)[0]])
                    bl = Blender(osp.join(r1,file1), osp.join(r2,file2), params)
                    processed_img_dict = bl.blend(args.blend_type)
                    os.system('mkdir -p %s' % fname)
                    for rho in processed_img_dict:
                        processed_img = processed_img_dict[rho]
                        processed_img = np.uint8(255*((processed_img - processed_img.min())/processed_img.ptp()))
                        plt.imsave(fname+'/'+str(rho)+'.jpg', processed_img)