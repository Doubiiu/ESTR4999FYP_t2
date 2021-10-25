import os
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
import torchvision
import math
import skimage.metrics
from tqdm import tqdm

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class TestVimeo90Kval:
    def __init__(self, file_list):
        self.transform = transforms.Compose([
            # transforms.Resize((256, 480)), # (h, w)
            transforms.ToTensor()
            ])
        with open(file_list, 'r') as f:
            lines = [line.rstrip('\n') for line in f.readlines()]
        self.triplet_file_list = ['data/sequences/'+f for f in lines]
        self.data_len = len(self.triplet_file_list)

        self.frame_f_list = []
        self.frame_l_list = []
        self.frame_gt_list = []

        for i in self.triplet_file_list:
            self.frame_f_list.append(to_variable(self.transform(Image.open(i + '/im1.png')).unsqueeze(0)))
            self.frame_l_list.append(to_variable(self.transform(Image.open(i + '/im3.png')).unsqueeze(0)))
            self.frame_gt_list.append(to_variable(self.transform(Image.open(i + '/im2.png')).unsqueeze(0)))

    def test_image(self, model, output_dir):
        sum_psnr = 0
        sum_ssim = 0
        pbar = tqdm(total=self.data_len, desc='', ascii=True, ncols=10)
        for i in range(len(self.triplet_file_list)):
            frame_i = model(self.frame_f_list[i], self.frame_l_list[i])
            #evaluate average psnr and ssim
            gt = self.frame_gt_list[i]
            #print('gt shape{} -- frame_i shape {}'.format(gt.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)).shape, frame_i.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)).shape))
            psnr = skimage.metrics.peak_signal_noise_ratio(gt.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), frame_i.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)))
            ssim = skimage.metrics.structural_similarity(gt.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), frame_i.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), multichannel=True)
            sum_psnr += psnr
            sum_ssim += ssim
            pbar.update(1)

        avg_psnr = sum_psnr / len(self.triplet_file_list)
        avg_ssim = sum_ssim / len(self.triplet_file_list)
        print('Completed! test {} triplets with average PSNR: {} and SSIM: {}'.format(len(self.triplet_file_list), avg_psnr, avg_ssim))
        pbar.close()
