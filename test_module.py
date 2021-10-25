import os
from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
import torchvision
import math
import skimage.metrics
import time


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class Test:
    def __init__(self, input_dir):
        self.dir_list = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
        # self.dir_list = ['Beanbags']
        self.transform = transforms.Compose([
            transforms.Resize((480, 640)), # (h, w)
            transforms.ToTensor()
            ])
        self.frame_f_list = []
        self.frame_l_list = []
        self.frame_gt_list = []

        for i in self.dir_list:
            self.frame_f_list.append(to_variable(self.transform(Image.open(input_dir + '/' + i + '/frame10.png')).unsqueeze(0)))
            self.frame_l_list.append(to_variable(self.transform(Image.open(input_dir + '/' + i + '/frame11.png')).unsqueeze(0)))
            self.frame_gt_list.append(to_variable(self.transform(Image.open(input_dir.replace('input', 'gt') + '/' + i + '/frame10i11.png')).unsqueeze(0)))

    def test_image(self, model, output_dir, num=12):
        sum_psnr = 0
        sum_ssim = 0
        frame_i = None
        for i in range(num):
            if not os.path.exists(output_dir + '/' + self.dir_list[i]):
                os.makedirs(output_dir + '/' + self.dir_list[i])
            if i == 0:
                torch.cuda.synchronize()
                start = time.time()
                frame_i = model(self.frame_f_list[i], self.frame_l_list[i])
                torch.cuda.synchronize()
                end = time.time()
                print('[MB-OTHER 480*640]inference time {:.2f}s'.format(end-start))
            else:
                frame_i = model(self.frame_f_list[i], self.frame_l_list[i])
            torchvision.utils.save_image(frame_i, output_dir + '/' + self.dir_list[i] + '/output.png', range=(0, 1))

            #evaluate average psnr and ssim
            gt = self.frame_gt_list[i]

            # print('gt shape{} -- frame_i shape {}'.format(gt.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)).shape, frame_i.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)).shape))
            psnr = skimage.metrics.peak_signal_noise_ratio(gt.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), frame_i.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)))
            ssim = skimage.metrics.structural_similarity(gt.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), frame_i.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), multichannel=True)
            sum_psnr += psnr
            sum_ssim += ssim
            #print('testing {}/{} images -- {} -- PSNR[{:2f}] SSIM[{:.4f}]'.format(i+1, len(self.dir_list), self.dir_list[i], psnr, ssim))
        avg_psnr = sum_psnr / num
        avg_ssim = sum_ssim / num
        print('[mb-other]Completed! test {} images with average PSNR: {} and SSIM: {}'.format(len(self.dir_list), avg_psnr, avg_ssim))
        return frame_i



class Middlebury_eval:
    def __init__(self, input_dir='./data/middlebury_eval'):
        self.im_list = ['Backyard', 'Basketball', 'Dumptruck', 'Evergreen', 'Mequon', 'Schefflera', 'Teddy', 'Urban']
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/input/' + item + '/frame10.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/input/' + item + '/frame11.png')).unsqueeze(0)))

    def test_image(self, model, output_dir='./evaluation/output', output_name='frame10i11.png'):
        model.eval()
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            # torchvision.utils.save_image(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))


class Davis:
    def __init__(self, input_dir='data/davis/input', gt_dir='data/davis/gt'):
        self.im_list = ['bike-trial', 'boxing', 'burnout', 'choreography', 'demolition', 'dive-in', 'dog-control', 'dolphins', 'e-bike', 'grass-chopper', 'hurdles', 'inflatable', 'juggle', 'kart-turn', 'kids-turning', 'lions', 'mbike-santa', 'monkeys', 'ocean-birds', 'pole-vault', 'running', 'selfie', 'skydive', 'speed-skating', 'swing-boy', 'tackle', 'turtle', 'varanus-tree', 'vietnam', 'wings-turn']
        self.transform = transforms.Compose([transforms.Resize((544, 960)), transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame10.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame11.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(gt_dir + '/' + item + '/frame10i11.png')).unsqueeze(0)))

    def test_image(self, model, output_dir='results/davis', logfile=None, output_name='output.png'):
        model.eval()
        sum_psnr = 0
        sum_ssim = 0

        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            if idx == 0:
                torch.cuda.synchronize()
                start = time.time()
                frame_i = model(self.input0_list[idx], self.input1_list[idx])
                torch.cuda.synchronize()
                end = time.time()
                print('[davis 544*960]inference time {:.2f}s'.format(end-start))
            else:
                frame_i = model(self.input0_list[idx], self.input1_list[idx])

            gt = self.gt_list[idx]
            # torchvision.utils.save_image(frame_i, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            psnr = skimage.metrics.peak_signal_noise_ratio(gt.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), frame_i.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)))
            ssim = skimage.metrics.structural_similarity(gt.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), frame_i.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), multichannel=True)
            sum_psnr += psnr
            sum_ssim += ssim

        avg_psnr = sum_psnr / len(self.im_list)
        avg_ssim = sum_ssim / len(self.im_list)
        print('[DAVIS]Completed! test {} images with average PSNR: {} and SSIM: {}'.format(len(self.im_list), avg_psnr, avg_ssim))




class ucf:
    def __init__(self, input_dir='data/ucf101'):
        self.im_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        self.transform = transforms.Compose([transforms.Resize((256, 320)),transforms.ToTensor()])

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            self.input0_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame0.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame2.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(input_dir + '/' + item + '/frame1.png')).unsqueeze(0)))

    def test_image(self, model, output_dir='', logfile=None, output_name='output.png'):
        model.eval()
        sum_psnr = 0
        sum_ssim = 0
        for idx in range(len(self.im_list)):
            if idx == 0:
                torch.cuda.synchronize()
                start = time.time()
                frame_i = model(self.input0_list[idx], self.input1_list[idx])
                torch.cuda.synchronize()
                end = time.time()
                print('[UCF101 256*320]inference time {:.2f}s'.format(end-start))
            else:
                frame_i = model(self.input0_list[idx], self.input1_list[idx])

            gt = self.gt_list[idx]
            psnr = skimage.metrics.peak_signal_noise_ratio(gt.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), frame_i.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)))
            ssim = skimage.metrics.structural_similarity(gt.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), frame_i.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), multichannel=True)
            sum_psnr += psnr
            sum_ssim += ssim

        avg_psnr = sum_psnr / len(self.im_list)
        avg_ssim = sum_ssim / len(self.im_list)
        print('[UCF101]Completed! test {} images with average PSNR: {} and SSIM: {}'.format(len(self.im_list), avg_psnr, avg_ssim))
