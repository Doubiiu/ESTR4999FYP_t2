import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import math

class VGGloss(torch.nn.Module):
    def __init__(self):
        super(VGGloss, self).__init__()

        model = torchvision.models.vgg19(pretrained=True)
        if torch.cuda.is_available():
            model = model.cuda()
        #VGG 19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512,(conv4_4 and relu4_4) 'M', 512, 512, 512, 512, 'M'],
        self.features = torch.nn.Sequential(
            #extract feature from relu4_4
            *list(model.features.children())[:-10]
        )
        # disable the grad
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, gt, output):
        gt_features = self.features(gt)
        out_features = self.features(output)
        return torch.norm(out_features-gt_features, 2).pow(2)

class SSIMloss(torch.nn.Module):
    '''

    '''
    def __init__(self, window_size=11, size_average=True, block=False):
        super(SSIMloss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.window = create_window(window_size)
        self.block = block

    def forward(self, im1, im2):
        if self.window.data.type() == im1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size)

            if im1.is_cuda:
                window = window.cuda(im1.get_device())
            window = window.type_as(im1)

            self.window = window
        if self.block:
            return _ssim(im1, im2, window, self.window_size, self.size_average, block=self.block)
        else:
            return -_ssim(im1, im2, window, self.window_size, self.size_average)


def create_window(window_size):
    sigma = 1.5
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    gaussian = gauss / gauss.sum()
    _1D_window = gaussian.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(3, 1, window_size, window_size).contiguous())
    return window

def _ssim(im1, im2, window, window_size, size_average=True, channel=3, block=False):
    mu1 = F.conv2d(im1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(im2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(im1 * im1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(im2 * im2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(im1 * im2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))


    if block:
        return (1-ssim_map[:, :, 0:64, 0:64].mean()).pow(3) + (1-ssim_map[:, :, 0:64, 64:128].mean()).pow(3) + (1-ssim_map[:, :, 0:64, 128:192].mean()).pow(3)+ (1-ssim_map[:, :, 0:64, 192:256].mean()).pow(3)+\
        (1-ssim_map[:, :, 64:128, 0:64].mean()).pow(3) + (1-ssim_map[:, :, 64:128, 64:128].mean()).pow(3) + (1-ssim_map[:, :, 64:128, 128:192].mean()).pow(3)+ (1-ssim_map[:, :, 64:128, 192:256].mean()).pow(3)+\
        (1-ssim_map[:, :, 128:192, 0:64].mean()).pow(3) + (1-ssim_map[:, :, 128:192, 64:128].mean()).pow(3) + (1-ssim_map[:, :, 128:192, 128:192].mean()).pow(3)+ (1-ssim_map[:, :, 128:192, 192:256].mean()).pow(3)+\
        (1-ssim_map[:, :, 192:256, 0:64].mean()).pow(3) + (1-ssim_map[:, :, 192:256, 64:128].mean()).pow(3) + (1-ssim_map[:, :, 192:256, 128:192].mean()).pow(3)+ (1-ssim_map[:, :, 192:256, 192:256].mean()).pow(3)
    else:
        return ssim_map.mean()
    # if size_average:
    #     return ssim_map.mean()
    # else:
    #     return ssim_map.mean(1).mean(1).mean(1)
