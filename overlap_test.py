from network import Overlap
import test_module
from dataset import TensorData
from args_list import args
from tqdm import tqdm
import skimage.metrics
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable

if __name__ == '__main__':
    model = Overlap()
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


        frame_i = model(frame_f, frame_l)
        psnr = skimage.metrics.peak_signal_noise_ratio(frame_g.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), frame_i.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)))
        ssim = skimage.metrics.structural_similarity(frame_g.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), frame_i.cpu().detach().squeeze(0).numpy().transpose((1, 2, 0)), multichannel=True)
        sum_psnr += psnr
        sum_ssim += ssim
        pbar.update(1)

    avg_psnr = sum_psnr / total_len
    avg_ssim = sum_ssim / total_len
    print('Completed! Vimeo test {} trplets with average PSNR: {} and SSIM: {}'.format(total_len, avg_psnr, avg_ssim))
    pbar.close()


    test_ = test_module.Test('./Interpolation_testset/input')
    #test
    test_.test_image(model, 'overlap')
    #for ucf test
    test_u = test_module.ucf()
    #test
    test_u.test_image(model)
    #for davis test
    test_d = test_module.Davis()
    #test
    test_d.test_image(model)
