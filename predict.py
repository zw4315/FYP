# -*- coding: utf-8 -*-
"""
Created on Wed May 15 18:04:01 2019

@author: 76590
"""

import os
import argparse
import torch
import json
import numpy
import glob
import torchvision.transforms as transforms
from torchvision.utils import save_image
from models import CompletionNetwork
from PIL import Image
from utils import poisson_blend, gen_input_mask_predict, specify_hole_area,gen_input_specific_mask


parser = argparse.ArgumentParser()



parser.add_argument('--max_holes', type=int, default=1)
parser.add_argument('--init_model_cn', type=str, default='1000') #50holes
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--hole_min_w', type=int, default=48)
parser.add_argument('--hole_max_w', type=int, default=96)
parser.add_argument('--hole_min_h', type=int, default=48)
parser.add_argument('--hole_max_h', type=int, default=96)


def main(args):

    args.model = os.path.expanduser('results/result/phase_3')
    args.config = os.path.expanduser('results/result/config.json')
    args.input_img = os.path.expanduser('datasets/shuju/test_sub/test_sss.jpg')
    
    if not os.path.exists(os.path.expanduser('results/Remove unwanted objects')):
        os.makedirs('results/Remove unwanted objects')
        print("The output path does not exist and is creating")
        
    args.output_img = os.path.expanduser('results/Remove unwanted objects/remove_test_spec.jpg')
    
        
    if args.init_model_cn == None:
        filename = args.model +'/*.png'
        filenames = glob.glob(filename)
        latest_model_idx = []
        for filename in filenames:
            latest_model_idx.append(int(filename[(filename.rfind("step")+4):-4]))      
            latest_state = max(latest_model_idx)
        latest_state = 100
        args.init_model_cn= 'model_cn_step' + str(latest_state)
    else:
        args.init_model_cn = 'model_cn_step' + args.init_model_cn
    args.model = os.path.expanduser(os.path.join( args.model,args.init_model_cn))
    
    print( "The completion model used here is ",  args.init_model_cn)
    
    # =============================================
    # Load model
    # =============================================
    with open(args.config, 'r') as f:
        config = json.load(f)
    mpv = config['mean_pv']
    model = CompletionNetwork()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))


    # =============================================
    # Predict
    # =============================================
    # convert img to tensor
    img = Image.open(args.input_img)
    img = transforms.Resize(args.img_size)(img)
    img = transforms.RandomCrop((args.img_size, args.img_size))(img)
    x = transforms.ToTensor()(img)
    x = torch.unsqueeze(x, dim=0)


    hole_area = specify_hole_area(50,80,80,80)
    
    mask_spec = gen_input_specific_mask(
        shape=x.shape,
        hole_area = hole_area,
        hole_size=(
            (args.hole_min_w, args.hole_max_w),
            (args.hole_min_h, args.hole_max_h),
        ),
        max_holes=args.max_holes,)
        
    print("The type of mask_spec is :", type(mask_spec))
    print("The shape of mask_spec is :", mask_spec.shape)
        
    msk_segmentation = gen_input_mask_predict(                         
        image_tensor=x,
        hole_size=args.hole_max_w
    )
    mask_spec = mask_spec.type(torch.FloatTensor)
    msk_segmentation = msk_segmentation.type(torch.FloatTensor)
    
    #msk = torch.mul(mask_spec, msk_segmentation)
    msk = msk_segmentation
    #print("What is the shape of MASK? ", msk.shape)

    #msk = msk_segmentation.type(torch.FloatTensor)  # this eliminate the effect of manual hole 
    # inpaint
    with torch.no_grad():
        input =  x - x * msk + mpv * msk
        output = model(input)
        inpainted = poisson_blend(input, output, msk)
        imgs = torch.cat((x, input, inpainted), dim=-1)
        save_image(imgs, args.output_img, nrow=3)
    print('output img was saved as %s.' % args.output_img)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
