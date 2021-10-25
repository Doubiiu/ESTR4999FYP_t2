import os
import argparse
import sys


parser = argparse.ArgumentParser(description='4999Video_frame_interpolation_by_ASepConv')
#total 55095 = train(51313) + val(3782)
parser.add_argument('--train_dataset_filelist', type=str, default='./data/tri_trainlist.txt', help='the dataset for training model')
parser.add_argument('--test_dataset_filelist', type=str, default='./data/tri_testlist.txt', help='the dataset for testing model')
parser.add_argument('--output_dir', type=str, default='./output', help='the output dir for storing checkpoint and results')
parser.add_argument('--kernel_size', type=int, default=3, help='hyper-parameter 1D kernel size')
parser.add_argument('--aug_data', action='store_true', help='if augmentation or not')
parser.add_argument('--model', type=str, default=None, help='model to be trained')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate when training model')
parser.add_argument('--loss', type=str, default='l1', help='type of training loss')
parser.add_argument('--epochs', type=int, default=50, help='epochs needed to train model')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size of data')
parser.add_argument('--test_input', type=str, default='./Interpolation_testset/input', help='input for showing real-time training result')
parser.add_argument('--test_gt', type=str, default='./Interpolation_testset/gt', help='groundtruth for showing real-time training result')
parser.add_argument('--val_type', type=int, default=0, help='get quantitative result on vimeo90K val dataset')

args = parser.parse_args()

if not os.path.exists(args.train_dataset_filelist):
    sys.exit("Error: dataset for training [{}] does't exist!".format(args.train_dataset_dir))
if not os.path.exists(args.output_dir):
    print("Note: output directory [{}] does't exist and will be created!".format(args.output_dir))
    os.makedirs(args.output_dir)
