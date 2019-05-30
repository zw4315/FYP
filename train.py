# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:29:57 2019

@author: 76590
"""
from tqdm import tqdm
from models import CompletionNetwork, ContextDiscriminator
from dataset import ImageDataset
from losses import completion_network_loss
from utils import (
    gen_input_mask,
    find_LB_corner_real,
    crop,
    crop_real,
    sample_random_batch,
    poisson_blend,
)
from torch.utils.data import DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import torch
import random
import glob
import os
import argparse
import numpy as np
import json


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='datasets/shuju')
parser.add_argument('--result_dir', type=str, default='results/result')
parser.add_argument('--init_model_cn', type=str, default='phase_3/model_cn_step')
parser.add_argument('--init_model_cd', type=str, default='phase_3/model_cd_step')
parser.add_argument('--config', type=str, default='config.json')
parser.add_argument('--steps_1', type=int, default=0)
parser.add_argument('--steps_2', type=int, default=0)
parser.add_argument('--steps_3', type=int, default=500)
parser.add_argument('--snaperiod_1', type=int, default=10)
parser.add_argument('--snaperiod_2', type=int, default=5)
parser.add_argument('--snaperiod_3', type=int, default=20)
parser.add_argument('--hole_min_w', type=int, default=48)
parser.add_argument('--hole_max_w', type=int, default=96)
parser.add_argument('--hole_min_h', type=int, default=48)
parser.add_argument('--hole_max_h', type=int, default=96)
parser.add_argument('--cn_input_size', type=int, default= 160)
parser.add_argument('--ld_input_size', type=int, default=96)
parser.add_argument('--optimizer', type=str, choices=['adadelta', 'adam'], default='adadelta')
parser.add_argument('--bsize', type=int, default=6)
parser.add_argument('--num_gpus', type=int, choices=[1, 2], default=1)
parser.add_argument('--alpha', type=float, default=4e-4)
parser.add_argument('--comp_mpv', default=True)
parser.add_argument('--max_mpv_samples', type=int, default=10000)

def main(args): 

    # ================================================
    # Preparation
    # ================================================
    
    # create result directory (if necessary)
    args.data_dir = os.path.expanduser(args.data_dir)
    args.result_dir = os.path.expanduser(args.result_dir)
    
    
    filename = args.result_dir +'/phase_3/*.png'
    filenames = glob.glob(filename)
    latest_model_idx = []
    for filename in filenames:
        latest_model_idx.append(int(filename[(filename.rfind("step")+4):-4]))
         
    init_state = max(latest_model_idx)
    print( "The latest version of model is ", init_state)
    
    if args.init_model_cn != None:
        args.init_model_cn = args.init_model_cn + str(init_state)
        args.init_model_cn = os.path.expanduser(os.path.join(args.result_dir,  args.init_model_cn))
        
      
    if args.init_model_cd != None:
        args.init_model_cd = args.init_model_cd + str(init_state)
        args.init_model_cd = os.path.expanduser(os.path.join(args.result_dir,  args.init_model_cd))
        
        
    if torch.cuda.is_available() == False:
        raise Exception('At least one gpu must be available.')
    if args.num_gpus == 1:
        # train models in a single gpu
        gpu_cn = torch.device('cuda:0')
        gpu_cd = gpu_cn
    else:
        # train models in different two gpus
        gpu_cn = torch.device('cuda:0')
        gpu_cd = torch.device('cuda:1')

#-----------------------------------------------------------------------------
    
    
    if os.path.exists(args.result_dir) == False:
        os.makedirs(args.result_dir)
       
    for s in ['phase_1', 'phase_2', 'phase_3']:
        if os.path.exists(os.path.join(args.result_dir, s)) == False:
            os.makedirs(os.path.join(args.result_dir, s))
            print('the path does not exist and is creating')

    # dataset
    trnsfm = transforms.Compose([
        transforms.Resize((args.cn_input_size)),
        transforms.RandomCrop((args.cn_input_size, args.cn_input_size)),
        transforms.ToTensor(),
    ])
    print('loading dataset... (it may take a few minutes)')
    train_dset = ImageDataset(os.path.join(args.data_dir, 'train'), trnsfm) ## the size is all images
    test_dset = ImageDataset(os.path.join(args.data_dir, 'test'), trnsfm)
    train_loader = DataLoader(train_dset, batch_size=args.bsize, shuffle=True)
    
    
    # =============================================
    # Load mean pixel value
    # =============================================
    if args.config != None:
        args.config = os.path.expanduser(os.path.join(args.result_dir, args.config))
        with open(args.config, 'r') as f:
            config = json.load(f)
        mean_pv = config['mean_pv']
        print( 'the mean pixel value of training images are loaded')
    
    else:
    # compute the mean pixel value of train dataset
        mean_pv = 0.
        imgpaths = train_dset.imgpaths[:min(args.max_mpv_samples, len(train_dset))]
        if args.comp_mpv:
            pbar = tqdm(total=len(imgpaths), desc='computing the mean pixel value')
            for imgpath in imgpaths:
                img = Image.open(imgpath)
                x = np.array(img, dtype=np.float32) / 255.
                mean_pv += x.mean()
                pbar.update()
            mean_pv /= len(imgpaths)
            pbar.close()
   
    
# save training config
    mpv = torch.tensor(mean_pv).to(gpu_cn)
    args_dict = vars(args)
    args_dict['mean_pv'] = mean_pv
    with open(args.config, mode='w') as f:
        json.dump(args_dict, f)
#-----------------------------------------------------------------------------        
 
    # ================================================
    # Training Phase 1
    # ================================================
    # model & optimizer
    model_cn = CompletionNetwork()
    if args.init_model_cn != None:
        model_cn.load_state_dict(torch.load(args.init_model_cn, map_location='cpu'))
        print("Initial model_cn is loaded")
    if args.optimizer == 'adadelta':
        opt_cn = Adadelta(model_cn.parameters())
    else:
        opt_cn = Adam(model_cn.parameters())
    model_cn = model_cn.to(gpu_cn)    #  send the netowrk to one gpu
    
    # training begins
    pbar = tqdm(total=args.steps_1)

    while pbar.n < args.steps_1:
        for x in train_loader:              #x.shape = [bsize,C,H,W]   for i, x in enumerate(train_loader):  
            opt_cn.zero_grad()     # To eliminate previous training record 
            msk, _ = gen_input_mask(                         
                image_tensor=x,
                hole_size=args.ld_input_size
            )
           
            msk = msk.type(torch.cuda.FloatTensor)
        
            
            # merge x, mask, and mpv
            msg = 'phase 1 |'
            x = x.to(gpu_cn)
            msk = msk.to(gpu_cn)
            input = x - x * msk + mpv * msk
            output = model_cn(input)

            # optimize
            loss = completion_network_loss(x, output, msk)
            loss.backward()
            opt_cn.step()

            msg += ' train loss: %.5f' % loss.cpu()
            pbar.set_description(msg)
            pbar.update()
            # test
            if pbar.n % args.snaperiod_1 == 0:
                with torch.no_grad():

                   x = sample_random_batch(test_dset, batch_size=args.bsize)
                   x = x.to(gpu_cn)
                   input = x - x * msk + mpv * msk
                   output = model_cn(input)
                   completed = poisson_blend(input, output, msk)
                   imgs = torch.cat((input.cpu(), completed.cpu()), dim=0)
                   save_image(imgs, os.path.join(args.result_dir, 'phase_1', 'step%d.png' % pbar.n), nrow=len(x))
                   torch.save(model_cn.state_dict(), os.path.join(args.result_dir, 'phase_1', 'model_cn_step%d' % pbar.n))

            if pbar.n >= args.steps_1:
                break        
    pbar.close()

    # ================================================
    # Training Phase 2
    # ================================================
    # model, optimizer & criterion
    model_cd = ContextDiscriminator(
        local_input_shape=(3, args.ld_input_size, args.ld_input_size),
        global_input_shape=(3, args.cn_input_size, args.cn_input_size),
    )
    if args.init_model_cd != None:
        model_cd.load_state_dict(torch.load(args.init_model_cd, map_location='cpu'))
        print("Initial model_cd is loaded")
    if args.optimizer == 'adadelta':
        opt_cd = Adadelta(model_cd.parameters())
    else:
        opt_cd = Adam(model_cd.parameters())
    model_cd = model_cd.to(gpu_cd)          # assign a model to a gpu?
    bceloss = BCELoss()

    # training
    pbar = tqdm(total=args.steps_2)
    while pbar.n < args.steps_2:
        for x in train_loader:

           
            opt_cd.zero_grad()

            # ================================================
            # fake
            # ================================================
         

            # create mask
              
            
            msk, corodinate = gen_input_mask(                         
                image_tensor=x,
                hole_size=args.ld_input_size
            )
           
            #print("the shape of coordinate array", len(corodinate) )
            x = x.to(gpu_cn)
            msk = msk.type(torch.cuda.FloatTensor)

            fake = torch.zeros((len(x), 1)).to(gpu_cd)
            msk = msk.to(gpu_cn)
            input_cn = x - x * msk + mpv * msk
            

            output_cn = model_cn(input_cn)

            input_gd_fake = output_cn.detach()
    
            input_ld_fake = crop(input_gd_fake, corodinate)
            input_fake = (input_ld_fake.to(gpu_cd), input_gd_fake.to(gpu_cd))
            output_fake = model_cd(input_fake)
            loss_fake = bceloss(output_fake, fake)

            # ================================================
            # real
            # ================================================
            hole_area_real =  find_LB_corner_real(
                size=(args.ld_input_size, args.ld_input_size),
                mask_size=(x.shape[3], x.shape[2]),
            )
            
            

            real = torch.ones((len(x), 1)).to(gpu_cd)
            input_gd_real = x
            input_ld_real = crop_real(input_gd_real, hole_area_real)
            input_real = (input_ld_real.to(gpu_cd), input_gd_real.to(gpu_cd))
            output_real = model_cd(input_real)
            loss_real = bceloss(output_real, real)

            # ================================================
            # optimize
            # ================================================
            loss = (loss_fake + loss_real) / 2.
            loss.backward()
            opt_cd.step()

            msg = 'phase 2 |'
            msg += ' train loss: %.5f' % loss.cpu()
            pbar.set_description(msg)
            pbar.update()

            # test
            if pbar.n % args.snaperiod_2 == 0:
                with torch.no_grad():

                    x = sample_random_batch(test_dset, batch_size=args.bsize)
                    x = x.to(gpu_cn)
                    input = x - x * msk + mpv * msk
                    output = model_cn(input)
                    completed = poisson_blend(input, output, msk)
                    imgs = torch.cat((input.cpu(), completed.cpu()), dim=0)
                    save_image(imgs, os.path.join(args.result_dir, 'phase_2', 'step%d.png' % pbar.n), nrow=len(x))
                    torch.save(model_cd.state_dict(), os.path.join(args.result_dir, 'phase_2', 'model_cd_step%d' % pbar.n))

            if pbar.n >= args.steps_2:
                break
    pbar.close()
   
    # ================================================
    # Training Phase 3
    # ================================================
    # training
    alpha = torch.tensor(args.alpha).to(gpu_cd)
    pbar = tqdm(total=args.steps_3)
    while pbar.n < args.steps_3:    
         for x in train_loader:
           
            
            # ================================================
            # train model_cd
            # ================================================
            opt_cd.zero_grad()
            
            # fake 
            # create mask
            msk, corodinate = gen_input_mask(                         
                image_tensor=x,
                hole_size=args.ld_input_size
            )
            
            x = x.to(gpu_cn)
            msk = msk.type(torch.cuda.FloatTensor)
            
            fake = torch.zeros((len(x), 1)).to(gpu_cd)
            msk = msk.to(gpu_cn)
            
            input_cn = x - x * msk + mpv * msk
            output_cn = model_cn(input_cn)
            
            input_gd_fake = output_cn.detach()
            input_ld_fake = crop(input_gd_fake, corodinate)
            
            input_fake = (input_ld_fake.to(gpu_cd), input_gd_fake.to(gpu_cd))
            output_fake = model_cd(input_fake)
            
            output_fake = model_cd(input_fake)
            loss_cd_1 = bceloss(output_fake, fake)
            
            # real
            
            hole_area_real =  find_LB_corner_real(
                size=(args.ld_input_size, args.ld_input_size),
                mask_size=(x.shape[3], x.shape[2]),
            )
            
            real = torch.ones((len(x), 1)).to(gpu_cd)
            input_gd_real = x
            input_ld_real = crop_real(input_gd_real, hole_area_real)
            input_real = (input_ld_real.to(gpu_cd), input_gd_real.to(gpu_cd))
            output_real = model_cd(input_real)
            loss_cd_2 = bceloss(output_real, real)
            
            # optimize
            loss_cd = (loss_cd_1 + loss_cd_2) * alpha / 2.
            loss_cd.backward()
            opt_cd.step()
            
            # ================================================
            # train model_cn
            # ================================================
            opt_cn.zero_grad()
            
            loss_cn_1 = completion_network_loss(x, output_cn, msk).to(gpu_cd)
            input_gd_fake = output_cn
            
            input_ld_fake = crop(input_gd_fake, corodinate)
            input_fake = (input_ld_fake.to(gpu_cd), input_gd_fake.to(gpu_cd))
            output_fake = model_cd(input_fake)
            loss_cn_2 = bceloss(output_fake, real)
            
            # optimize
            loss_cn = (loss_cn_1 + alpha * loss_cn_2) / 2.
            loss_cn.backward()
            opt_cn.step()

            msg = 'phase 3 |'
            msg += ' train loss (cd): %.5f' % loss_cd.cpu()
            msg += ' train loss (cn): %.5f' % loss_cn.cpu()
            pbar.set_description(msg)
            pbar.update()

            # test
            if pbar.n % args.snaperiod_3 == 0:
                with torch.no_grad():

                    x = sample_random_batch(test_dset, batch_size=args.bsize)
                    x = x.to(gpu_cn)
                    input = x - x * msk + mpv * msk
                    output = model_cn(input)
                    completed = poisson_blend(input, output, msk)
                    imgs = torch.cat((input.cpu(), completed.cpu()), dim=0)
                    model_index = pbar.n+init_state
                    save_image(imgs, os.path.join(args.result_dir, 'phase_3', 'step%d.png' % model_index), nrow=len(x))
                    torch.save(model_cn.state_dict(), os.path.join(args.result_dir, 'phase_3', 'model_cn_step%d' % model_index))
                    torch.save(model_cd.state_dict(), os.path.join(args.result_dir, 'phase_3', 'model_cd_step%d' % model_index))

            if pbar.n >= args.steps_3:
                break
    pbar.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
