from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
from torchvision.transforms import CenterCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
import random


def default_loader(path):
    return Image.open(path)

class TensorData(Dataset):

    def __init__(self, filelist, aug_data, train=True, video=False):
        super(TensorData, self).__init__()
        if aug_data:
            self.transform = transforms.Compose([
                # RandomRotation((90, 90)),
                RandomVerticalFlip(1.0),
                RandomHorizontalFlip(1.0),
                # transforms.Resize((128, 128)),
                transforms.ToTensor()
            ])
        else:
            if train:
                self.transform = transforms.Compose([
                    transforms.ToTensor()
                ])
            else:
                if video:
                    self.transform = transforms.Compose([
                        transforms.Resize((736, 1280)), # (h, w)
                        transforms.ToTensor()
                    ])
                else:
                    self.transform = transforms.Compose([
                        transforms.Resize((256, 480)), # (h, w)
                        transforms.ToTensor()
                    ])
        with open(filelist, 'r') as f:
            lines = [line.rstrip('\n') for line in f.readlines()]
        if train:
            self.triplet_file_list = ['data/sequences_crop/'+f for f in lines]
        else:
            if video:
                self.triplet_file_list = ['data/video_frames_720p/'+f for f in lines]
            else:
                self.triplet_file_list = ['data/sequences/'+f for f in lines]
        self.data_len = len(self.triplet_file_list)
        self.train = train


    @staticmethod
    def random_swap(frame_1, frame_3):
        if random.random() > 0.5:
            return frame_1, frame_3
        else:
            return frame_3, frame_1

    def get_file_list(self):
        return self.triplet_file_list

    def __getitem__(self, index):
        frame_f = self.transform(default_loader(self.triplet_file_list[index] + "/im1.png"))
        frame_g = self.transform(default_loader(self.triplet_file_list[index] + "/im2.png"))
        frame_l = self.transform(default_loader(self.triplet_file_list[index] + "/im3.png"))
        if self.train:
            frame_f, frame_l = self.random_swap(frame_f, frame_l)
        return frame_f, frame_g, frame_l

    def __len__(self):
        return self.data_len
