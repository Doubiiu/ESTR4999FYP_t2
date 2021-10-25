import torch
import os
import pynvml
import copy
import sys

from args_list import args
from dataset import TensorData
from network_dcn_resblock import IASC
from loss import VGGloss
from loss import SSIMloss
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
from utilities.utils import *
import test_module

#path to save weights
weights_save_dir = args.output_dir + '/weights_'+str(args.kernel_size)+'_'+args.loss+'_'+str(args.epochs)
if not os.path.exists(weights_save_dir):
    os.makedirs(weights_save_dir)

#set hyper-parameters
start_epoch = 0
checkpoint = None
epochs = args.epochs
batch_size = args.batch_size
if args.loss == 'l1':
    loss_cal = torch.nn.L1Loss()
elif args.loss == 'vgg':
    loss_cal = VGGloss()
elif args.loss == 'ssim':
    loss_cal = SSIMloss()
elif args.loss == 'ssim_b':
    loss_cal = SSIMloss(block=True)
else:
    raise ValueError('Unknown type of loss: {}'.format(args.loss))

print("################################")
print("Loading model")
if args.model == None:
    lr = args.lr
    kernel_size = args.kernel_size
    model = IASC(kernel_size, lr)
else:
    checkpoint = torch.load(args.model)
    kernel_size = checkpoint['kernel_size']
    start_epoch = checkpoint['epoch'] + 1
    model = IASC(kernel_size)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

if torch.cuda.is_available():
    model = model.cuda()

optimizer = optim.Adamax(model.parameters(), lr=args.lr)
if checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


print("################################")
print("Loading data")
num_workers = 4
if os.name == "nt":
    num_workers = 0
train_set = TensorData(args.train_dataset_filelist, args.aug_data)
val_set = TensorData(args.test_dataset_filelist, False, False, False)
train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
total_train_step = train_loader.__len__()
# val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers)
# total_val_step = val_loader.__len__()


print("################################")
print("Start training with {}".format('GPU' if torch.cuda.is_available() else 'CPU'))
config_txt = 'kernel_size: {}\naug_data:{}\nlr:{}\nloss:{}\nepochs:{}\nbatch_size:{}'.\
format(args.kernel_size, args.aug_data, args.lr, args.loss, args.epochs, args.batch_size)
print(config_txt)
with open(weights_save_dir+'/config.txt', 'w') as f:
    f.write(config_txt)
writer = SummaryWriter(comment='log') # Visualize the training loss
record = {
    "cur": 0,
    "best": sys.float_info.max,
    "epoch": 0,
    "model": None,
    "optim": None
}
# Display GPU info:
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

def train(epoch, data_loader):
    model.train()
    loss_metric = AvgMetric(len(data_loader))
    info = 'Epoch '+str(epoch+1)+' Mem: {:.1f}/{:.1f}'.format(meminfo.used/1024**3, meminfo.total/1024**3)
    pbar = tqdm(total=total_train_step, desc=info, ascii=True, ncols=10)
    for idx, (frame_f, frame_g, frame_l) in enumerate(data_loader):
        if torch.cuda.is_available():
            frame_f = Variable(frame_f.cuda())
            frame_g = Variable(frame_g.cuda())
            frame_l = Variable(frame_l.cuda())
        else:
            frame_f = Variable(frame_f)
            frame_g = Variable(frame_g)
            frame_l = Variable(frame_l)

        optimizer.zero_grad()
        output = model(frame_f, frame_l)
        loss = loss_cal(frame_g, output)
        loss_metric.add(loss.item())
        loss.backward()
        optimizer.step()

        # Visulaization of training result
        pbar.update(1)
        niter = epoch * total_train_step + idx
        writer.add_scalar('Train', loss_metric.average(idx+1), niter)
    print("[Train] Epoch {}/{} complete: Avg. Loss={:.4f}".format(epoch+1, epochs,loss_metric.average()))
    pbar.close()

def val(epoch, data_loader=1):
    model.eval()
    # test to see the result
    test_ = test_module.Test(args.test_input)
    im = test_.test_image(model, args.output_dir + '/test_result_'+str(args.kernel_size)+'_'+args.loss+'_'+str(epochs)+'/' + str(epoch), 1)
    writer.add_image('Beanbags', im.squeeze(0), epoch)
    # loss_metric = AvgMetric(len(data_loader))
    # with torch.no_grad():
    #     for idx, (frame_f, frame_g, frame_l) in enumerate(data_loader):
    #         if torch.cuda.is_available():
    #             frame_f = Variable(frame_f.cuda())
    #             frame_g = Variable(frame_g.cuda())
    #             frame_l = Variable(frame_l.cuda())
    #         else:
    #             frame_f = Variable(frame_f)
    #             frame_g = Variable(frame_g)
    #             frame_l = Variable(frame_l)
    #
    #         output = model(frame_f, frame_l)
    #         loss = loss_cal(frame_g, output)
    #         loss_metric.add(loss.item())
    #         niter = epoch * total_val_step + idx
    #         writer.add_scalar('val', loss_metric.average(idx+1), niter)
    #     record['cur'] = loss_metric.average()
    #     if record['cur'] < record['best']:
    #         record['best'] = record['cur']
    #         record['epoch'] = epoch
    #         record['model'] = copy.deepcopy(model.state_dict())
    #         record['optim'] = copy.deepcopy(optimizer.state_dict())
    #     print("[Test] Epoch {cur}/{total} complete: Avg.loss={val:.4f}".format(cur=epoch+1, total=epochs,
    #                                                                             val=loss_metric.average()))

# train-val iter
for epoch in range(start_epoch, epochs):
    train(epoch, train_loader)
    val(epoch)
    if (epoch + 1) % 1 == 0:
        torch.save(
            {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                'kernel_size': kernel_size,
            },
            weights_save_dir + '/model_epoch' + str(epoch+1).rjust(3, '0') + '.pytorch'
         )

writer.close()
# save best model
if record['model'] != None:
    print('Best result with loss: {:.4f}'.format(record['best']))
    torch.save(
        {
            'state_dict': model.state_dict(),
            'epoch': record['epoch'],
            "optimizer_state_dict": record['optim'],
            'kernel_size': kernel_size,
        },
        weights_save_dir + '/model_best_' + str(record['epoch']) + '.pytorch'
     )
