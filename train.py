#!/usr/bin/python3
#coding=utf-8

import sys
import datetime
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from net  import SaliencyDisentangle
import pytorch_ssim

# Loss Configureation
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)

def floss(prediction, target, beta=0.3, log_like=False):
        EPS = 1e-10
        prediction  = torch.sigmoid(prediction)
        batch_size=prediction.size(0)
        fmatrix=torch.zeros(batch_size,1).cuda()
        for i in range(batch_size):      
            N = N = prediction[i:i+1,:,:].size(1)
            TP = (prediction[i:i+1,:,:] * target[i:i+1,:,:]).view(N, -1).sum(dim=1)
            H = beta * target[i:i+1,:,:].view(N, -1).sum(dim=1) + prediction[i:i+1,:,:].view(N, -1).sum(dim=1)
            fmatrix[i] = (1 + beta) * TP / (H + EPS)
        fmeasure=torch.mean(fmatrix).cuda()
        if log_like:
            floss = -torch.log(fmeasure)
        else:
            floss  = (1 - fmeasure).cuda()
        return floss
    
def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()


def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='/data/DUTS-TR', savepath='./out', mode='train', batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=50)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True, num_workers=8)
    ## network
    net    = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer      = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw             = SummaryWriter(cfg.savepath)
    global_step    = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr*0.1
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, mask, body, detail) in enumerate(loader):
            image, mask, body, detail = image.cuda(), mask.cuda(), body.cuda(), detail.cuda()
            out_detail, out_label = net(image)

            loss_detail = F.binary_cross_entropy_with_logits(out_detail, body) + ssim_loss(out_detail, detail)
            loss_label = F.binary_cross_entropy_with_logits(out_label, mask) + iou_loss(out_label, mask)+ floss(out_label, mask)
            loss   = (loss_detail + loss_label)/2

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss_detail':loss_detail.item(), 'loss_label':loss_label.item()}, global_step=global_step)
            if step%10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss_detail=%.3f | loss_label=%.3f'
                    %(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss_detail.item(), loss_label.item()))

        if epoch > cfg.epoch*3/4:
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))


if __name__=='__main__':
    train(dataset, SaliencyDisentangle)