import os
import cv2
import torch
import PWCNet.run
import PIL
import PIL.Image
import numpy as np
from tqdm import tqdm
import random
import time
'''
    This file is to do the data preprocessing for interpolation.
    It aims to get patches(150*150) from original frames(448*256)
        according to the magnitude of optical flow between first and last frame
矢量方向代表色相
矢量长度代表饱和度

'''
class Data_prepare():
    def __init__(self, filelist, output_dir):
        # prepare data for training
        # original dataset size: 32.2G
        with open(filelist, 'r') as f:
            lines = [line.rstrip('\n') for line in f.readlines()]
        self.triplet_file_list = ['data/sequences/'+f for f in lines]
        self.triplet_file_list = self.triplet_file_list[41900:]
        self.filelist = filelist
        self.output_dir = output_dir

    def data_len(self):
        return len(self.triplet_file_list)

    def random_crop_and_save(self):
        pbar = tqdm(total=len(self.triplet_file_list), desc='', ascii=True, ncols=10)
        for image_path in self.triplet_file_list:
            img1 = PIL.Image.open(image_path + '/im1.png')
            img2 = PIL.Image.open(image_path + '/im2.png')
            img3 = PIL.Image.open(image_path + '/im3.png')
            width, height = img1.size
            assert(height == 256)
            assert(width == 448)
            max_mean_of = 0
            selected_cropped1 = None
            selected_cropped2 = None
            selected_cropped3 = None
            for i in range(4):
                # r = random.randInt(0, height-width)
                r = 48*i
                cropped1 = img1.crop((r, 0, r+height, height))
                cropped2 = img2.crop((r, 0, r+height, height))
                cropped3 = img3.crop((r, 0, r+height, height))

                of = self.check_optical_flow(cropped1, cropped3)
                if  of > max_mean_of:
                    max_mean_of = of
                    selected_cropped1 = cropped1
                    selected_cropped2 = cropped2
                    selected_cropped3 = cropped3
            save_image_path = image_path.replace('sequences', self.output_dir)

            if not os.path.exists(save_image_path):
                try:
                    os.mkdir(save_image_path)
                except OSError:
                    pass
            selected_cropped1.save(save_image_path + '/im1.png')
            selected_cropped2.save(save_image_path + '/im2.png')
            selected_cropped3.save(save_image_path + '/im3.png')
            pbar.update(1)
        pbar.close()

    @staticmethod
    def check_optical_flow(first_image, last_image):
        tensorFirst = torch.FloatTensor(np.array(first_image)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
        tensorSecond = torch.FloatTensor(np.array(last_image)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
        tensorOutput = PWCNet.run.estimate(tensorFirst, tensorSecond)

        flowmatrix = np.array(tensorOutput.numpy().transpose(1, 2, 0), np.float32)
        mag, _ = cv2.cartToPolar(flowmatrix[..., 0], flowmatrix[..., 1])
        # print(mag.shape)
        # print('min:{}, max:{}, mean:{}'.format(np.min(mag), np.max(mag), np.mean(mag)))
        return np.mean(mag)



if __name__ == '__main__':
    start = time.time()
    data_prepare = Data_prepare('data/tri_trainlist.txt', 'sequences_crop')
    data_prepare.random_crop_and_save()
    print('Data preparation for {} images completed in {:.2f} seconds'.format(data_prepare.data_len(), time.time()-start))
