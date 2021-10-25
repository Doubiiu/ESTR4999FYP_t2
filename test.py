import torch
from args_list import args
import os
from test_module import Test
from test_module import Davis
from test_module import ucf
from vimeo_val import TestVimeo90Kval
from dataset import TensorData
from torch.utils.data import DataLoader

from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import torchvision
import skimage.metrics
from tqdm import tqdm
import pynvml
import time
from network_dcn_whole import IASC


def test(num):
    test_input = args.test_input
    # test_gt = args.test_gt
    model_path = args.output_dir + '/weights_5_vgg_100/model_epoch0'+str(num)+'.pytorch'
    test_result = args.output_dir + '/test_result_5_vgg_100/'+str(num-1)
    print('Testing model: ' + model_path)
    if not os.path.exists(test_result):
        os.makedirs(test_result)

    #load model
    checkpoint = torch.load(model_path)
    model = IASC(checkpoint['kernel_size'])
    model.load_state_dict(checkpoint['state_dict'])
    #model parameters stat.
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    if torch.cuda.is_available():
        model = model.cuda()
    #load data
    if args.val_type == 0:

        #for middlebury test
        test_ = Test(test_input)
        #test
        test_.test_image(model, test_result)
        #for ucf test
        test_u = ucf()
        #test
        test_u.test_image(model)
        #for davis test
        test_d = Davis()
        #test
        test_d.test_image(model)
        #for vimeo90K val benchmark
        data = TensorData(args.test_dataset_filelist, aug_data=False, train=False)
        data_loader = DataLoader(data, 1, num_workers=0)
        total_len = data_loader.__len__()
        sum_psnr = 0
        sum_ssim = 0
        pbar = tqdm(total=total_len, desc='', ascii=True, ncols=10)
        for idx, (frame_f, frame_g, frame_l) in enumerate(data_loader):
            if torch.cuda.is_available():
                frame_f = Variable(frame_f.cuda())
                frame_g = Variable(frame_g.cuda())
                frame_l = Variable(frame_l.cuda())
            else:
                frame_f = Variable(frame_f)
                frame_g = Variable(frame_g)
                frame_l = Variable(frame_l)

            if idx == 0:
                torch.cuda.synchronize()
                start = time.time()
                frame_i = model(frame_f, frame_l)
                torch.cuda.synchronize()
                end = time.time()
                print('[Vimeo90K 256*480]inference time {:.2f}s'.format(end-start))
            else:
                frame_i = model(frame_f, frame_l)

            #evaluate average psnr and ssim
            #print('gt shape{} -- frame_i shape {}'.format(gt.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)).shape, frame_i.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)).shape))
            psnr = skimage.metrics.peak_signal_noise_ratio(frame_g.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), frame_i.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)))
            ssim = skimage.metrics.structural_similarity(frame_g.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), frame_i.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), multichannel=True)
            sum_psnr += psnr
            sum_ssim += ssim
            pbar.update(1)

        avg_psnr = sum_psnr / total_len
        avg_ssim = sum_ssim / total_len
        print('[Vimeo90K]Completed! test {} trplets with average PSNR: {} and SSIM: {}'.format(total_len, avg_psnr, avg_ssim))
        pbar.close()

    elif args.val_type == 2:
        #for video interpolation
        model.eval()
        data = TensorData('data/tri_videoframes.txt', aug_data=False, train=False, video=True)
        data_loader = DataLoader(data, 1, num_workers=0)
        total_len = data_loader.__len__()
        sum_psnr = 0
        sum_ssim = 0
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = 0

        pbar = tqdm(total=total_len, desc='', ascii=True, ncols=10)
        for idx, (frame_f, frame_g, frame_l) in enumerate(data_loader):
            if torch.cuda.is_available():
                frame_f = Variable(frame_f.cuda())
                frame_g = Variable(frame_g.cuda())
                frame_l = Variable(frame_l.cuda())
            else:
                frame_f = Variable(frame_f)
                frame_g = Variable(frame_g)
                frame_l = Variable(frame_l)
            frame_i = model(frame_f, frame_l)

            if idx == 0:
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                info = 'Mem: {:.1f}/{:.1f}'.format(meminfo.used/1024**3, meminfo.total/1024**3)
                print('\n'+info+'\n')
            #evaluate average psnr and ssim
            torchvision.utils.save_image(frame_i, data.get_file_list()[idx] +'/imi_'+model_path.split('/')[-2]+'_'+model_path.split('/')[-1].split('.')[0]+'.png', range=(0, 1))
            psnr = skimage.metrics.peak_signal_noise_ratio(frame_g.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), frame_i.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)))
            ssim = skimage.metrics.structural_similarity(frame_g.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), frame_i.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), multichannel=True)
            sum_psnr += psnr
            sum_ssim += ssim
            pbar.update(1)

        avg_psnr = sum_psnr / total_len
        avg_ssim = sum_ssim / total_len
        print('Completed! test {} trplets with average PSNR: {} and SSIM: {}'.format(total_len, avg_psnr, avg_ssim))
        pbar.close()



    else:
        raise ValueError('Unknown type of val: {}'.format(args.val_type))


if __name__ == "__main__":
    for i in range(80, 81):
        test(i)
