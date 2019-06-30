import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import numpy as np
from poissonblending import blend
import os
import sys
import random
import math
import skimage.io
import matplotlib
import matplotlib.pyplot as plt


Msk_ROOT_DIR = os.path.abspath('mrcnn')
sys.path.append(Msk_ROOT_DIR)  # To find local version of the maskrcnn library

from mrcnn import mskutils  
import mrcnn.model as modellib
from mrcnn import visualize
sys.path.append(os.path.join(Msk_ROOT_DIR, "samples/coco/"))  # To find local version

import coco

#-------------------------------------------------------------------------------------------
COCO_MODEL_PATH = os.path.join(Msk_ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
    mskutils.download_trained_weights(COCO_MODEL_PATH)

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

if not os.path.exists(os.path.join(Msk_ROOT_DIR, "logs")):
    os.makedirs(os.path.join(Msk_ROOT_DIR, "logs"))

#-------------------------------------------------------------------------------------------
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
#-------------------------------------------------------------------------------------------

# Create model object in inference mode and load trained weight
MODEL_DIR = os.path.join(Msk_ROOT_DIR, "logs")
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)   
#---------------------------------------------------------------------------
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
#---------------------------------------------------------------------------
def select_mask(mask, hole_size):
  """
        * input:
            - hole_size:
                The length of local area of the image, it should cover the whole area of the graph.
            - mask:
                It is an array containing the all masks of instances shwon in one image.
        * returns:
            - rand_instance_msk: 
                A mask of a random-selected instance.
            - corodinate:
                The coordinate of the left bottom corner of the local area
  """

  idx = mask.shape[0] -1 
  oversize = True
  iter_amount=0
  while oversize == True:     
      mask_copy=mask[random.randint(0,idx)]   # randomly select an instance with mask
      iter_amount=iter_amount+1
   
      
      non_zero_ele = torch.nonzero(mask_copy,out=None)
  
      
      max_boundary = torch.max(non_zero_ele,dim=0)[0] 
      min_boundary = torch.min(non_zero_ele,dim=0)[0] 
  
      maxlen = torch.max(max_boundary - min_boundary)   #the size of single instnace mask in single graph 
      
      
  
      
      if maxlen <= hole_size:
         
          corodinate = find_LB_corner(max_boundary, min_boundary, mask_copy.shape[0], hole_size )
          rand_instance_msk = np.stack((mask_copy,)*3, axis=0)
          rand_instance_msk = rand_instance_msk.astype('uint8')
          oversize=False
    
      if iter_amount>idx+1:
          #print("Oversized, so randomly generate the mask")
          manual_mask =torch.zeros(size = mask_copy.shape)    #desired torch.Size([3, 160, 160])         
          hole_w = random.randint(1,hole_size)
          hole_h = random.randint(1,hole_size)
          offset_x = random.randint(0,mask_copy.shape[0] - hole_w)
          offset_y = random.randint(0,mask_copy.shape[1] - hole_h)
          
          manual_mask[offset_y : offset_y+hole_h, offset_x : offset_x+hole_w] = 1.0
          non_zero_ele = torch.nonzero(manual_mask,out=None)
          max_boundary = torch.max(non_zero_ele,dim=0)[0] 
          min_boundary = torch.min(non_zero_ele,dim=0)[0] 
          maxlen = torch.max(max_boundary - min_boundary)   #the size of single instnace mask in single graph 
      
          corodinate = find_LB_corner(max_boundary, min_boundary, mask_copy.shape[0], hole_size )
       
          
          
         
          
          rand_instance_msk = np.stack((manual_mask,)*3, axis=0)
          rand_instance_msk = rand_instance_msk.astype('uint8')
          oversize=False
          
          
      
  return rand_instance_msk, corodinate 

#------------------------------------------------------------------------------
def segment_predict(mask, index, expend): 
  """
        * input:
            - index:
                The index of the relevent mask shwon on the image.
            - mask:
                It is an array containing all relevent masks of instances shwon in one image.
        * returns:
            - selected_instance_msk: 
                A mask of selected instance.
    
  """ 
  expend = int(expend) +1  
  index = int(index) -1 
  mask_copy=mask[index]   # select an instance with mask
  expand_msk = []
  

  
  for row in range(mask_copy.shape[0]-expend):
      for col in range(mask_copy.shape[1]-expend):
          if (mask_copy[row][col] *mask_copy[row][col] !=  mask_copy[row][col+1] *mask_copy[row][col+1]):
             for ex_pixel in range(0, expend):
                 expand_msk.append((row,col+ ex_pixel))
                 if col>ex_pixel: expand_msk.append((row,col-ex_pixel))
                 
             
              
          if (mask_copy[row][col] *mask_copy[row][col] !=  mask_copy[row+1][col] *mask_copy[row+1][col]):
             for ex_pixel in range(0, expend):
                 expand_msk.append((row+ex_pixel,col))
                 if row>ex_pixel: expand_msk.append((row-ex_pixel,col))
       
  expand_msk = set(expand_msk)
  for element in expand_msk:
     # print("What is element in the list? ", element)
      i,j= element[0],element[1]
      mask_copy[i][j] = 1
 # for col in range(mask_copy.shape[1]-1):
 #    for row in range(mask_copy.shape[0]-1):
  #       if mask_copy[row][col] *mask_copy[row][col] !=  mask_copy[row+1][col] *mask_copy[row+1][col]:
   #           mask_copy[row][col] = 1
    #          mask_copy[row+1][col] =1        
              
  
  selected_instance_msk = np.stack((mask_copy,)*3, axis=0)
  selected_instance_msk = selected_instance_msk.astype('uint8')
  return selected_instance_msk 
     
#---------------------------------------------------------------------------
def gen_input_mask(
    image_tensor,
    hole_size):
    """
    * inputs:
        - image_tensor x in one batch 
                A 4D tuple (samples, c, h, w) is assumed.
        - hole_size
                The size of the expected completion region.
       
    * returns:
            Input mask tensor with holes.
            All the pixel values within holes are filled with 1,
            while the other pixel values are 0.
    """
   
    
    trnsfm = transforms.Compose([
        transforms.ToTensor(),
        ])
    images = []                  #store the images in one batch
    mask_tensors = []             #store the masks in one batch

    for item in range(image_tensor.shape[0]):
        
        single_image = image_tensor[item]
    
        PILimg = 255*single_image.numpy()
       
        
        PILimg=PILimg.transpose(1,2,0).astype(np.uint8)
        images.append(PILimg)
    
  
    for img in range(len(images)):
        image = images[img]
        results = model.detect([image], verbose=0)
        r = results[0]
 
        img_shape = image.shape[0]                        #(160, 160, 3)
        
        
        
        Instance = r['rois'].shape[0]
        mask = torch.zeros(size = (img_shape,img_shape,1))
    
        
        if not Instance:
            #print("Dear, there is no instance and love you")
            mask = torch.zeros(size = (img_shape,img_shape))
            hole_w = random.randint(1,0.5*hole_size)
            hole_h = random.randint(1,0.5*hole_size)
            offset_x = random.randint(0,img_shape - hole_w)
            offset_y = random.randint(0,img_shape - hole_h)
            mask[offset_y : offset_y+hole_h, offset_x : offset_x+hole_w] = 1.0
            mask = mask.unsqueeze(0)
            mask_tensors.append(mask)
            
            
        
        else:    
            #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']) 
            mask_tensor = trnsfm(r['masks'].astype(int))    # mask tensor for single image
            mask_tensors.append(mask_tensor)                  #store masks for all images in a batch
        
        
    
    output_masks = []
    corodinate_arr = []
    #sig_mask, one image with all masks
    for sig_mask in range(len(mask_tensors)):                # stack the mask of each image into a tensor
        tensor_msk, corodinate = select_mask(mask_tensors[sig_mask],hole_size=hole_size)
        tensor_msk=torch.from_numpy(tensor_msk) # convert ndarray to tensor, 
        output_masks.append(tensor_msk)
        corodinate_arr.append(corodinate)
     
    stacked_msk_tensor = torch.stack(output_masks)   
    
    return stacked_msk_tensor, corodinate_arr      #msk tensors for different images in one batch

#---------------------------------------------------------------------------------------------------------
## This is the function only used in testing image 
def gen_input_specific_mask(
    shape, hole_size,
    hole_area=None, max_holes=1):
    """
    * inputs:
        - shape (sequence, required):
                Shape of output mask.
                A 4D tuple (samples, c, h, w) is assumed.
        - hole_size (sequence or int, required):
                Size of holes created in a mask.
                If a sequence of length 4 provided,
                holes of size (w, h) = (
                    hole_size[0][0] <= hole_size[0][1],
                    hole_size[1][0] <= hole_size[1][1],
                ) are generated.
                All the pixel values within holes are filled with 1.
        - hole_area (sequence, optional):
                This argument constraints the area where holes are generated.
                hole_area[0] is the left corner (x, y) of the area,
                while hole_area[1] is its width and height (w, h).
                This area is used as the input region of Local discriminator.
                The default value is None.
        - max_holes (int, optional):
                This argument specifies how many holes are generated.
                The number of holes is randomly chosen from [1, max_holes].
                The default value is 1.
    * returns:
            Input mask tensor with holes.
            All the pixel values within holes are filled with 1,
            while the other pixel values are 0.
    """
    mask = torch.zeros(shape)
    bsize, _, mask_h, mask_w = mask.shape
    masks = []
    for i in range(bsize):
        n_holes = random.choice(list(range(1, max_holes+1)))
        for j in range(n_holes):
            # choose patch width
            if isinstance(hole_size[0], tuple) and len(hole_size[0]) == 2:
                hole_w = random.randint(hole_size[0][0], hole_size[0][1])
            else:
                hole_w = hole_size[0]

            # choose patch height
            if isinstance(hole_size[1], tuple) and len(hole_size[1]) == 2:
                hole_h = random.randint(hole_size[1][0], hole_size[1][1])
            else:
                hole_h = hole_size[1]

            # choose offset upper-left coordinate
            if hole_area:
                #harea_xmin, harea_ymin = hole_area[0]
                offset_x, offset_y = hole_area[0]
                hole_w, hole_h = hole_area[1]
                #offset_x = random.randint(harea_xmin, harea_xmin + harea_w - hole_w)
                #offset_y = random.randint(harea_ymin, harea_ymin + harea_h - hole_h)
            else:
                offset_x = random.randint(0, mask_w - hole_w)
                offset_y = random.randint(0, mask_h - hole_h)
            mask[i, :, offset_y : offset_y + hole_h, offset_x : offset_x + hole_w] = 1.0
    return mask    




#------------------------------------------------------------------------------------------
def specify_hole_area(offset_x, offset_y, harea_w, harea_h):
    """
    * inputs:
        - The coordinate
        - width and height
    * returns:
            A sequence which is used for the input argument 'hole_area' of function 'gen_input_mask'.
    """
    return ((offset_x, offset_y), (harea_w, harea_h))    

    
def gen_input_mask_predict(
    image_tensor,
    hole_size):   
    """
    Return msk tensors for different images in one batch
    
    """
    trnsfm = transforms.Compose([
    transforms.ToTensor(),
    ])
    
    images = []                  #store the images in one batch
    desire_msk_tensors = []             #store the desirable masks in one batch
   

    for item in range(image_tensor.shape[0]):
        
        single_image = image_tensor[item]
        PILimg = 255*single_image.numpy()
        PILimg=PILimg.transpose(1,2,0).astype(np.uint8)
        images.append(PILimg)
    
  
    for img in range(len(images)):
        aim_msk_sigimg = []
        
        image = images[img]
        results = model.detect([image], verbose=0)
        r = results[0]
        img_shape = image.shape[0]                        #(160, 160, 3)
        sig_imgname = []                 # contain the name of all instances in a single image 
        instance_idx = []                   #The index of instance in one image
        
        Instance = r['rois'].shape[0]
        
        

        if not Instance:
            print("Dear, No instance was detected in the testing image so randomly cropped")
            mask = torch.zeros(size = (img_shape,img_shape))
            hole_w = random.randint(1,0.5*hole_size)
            hole_h = random.randint(1,0.5*hole_size)
            offset_x = random.randint(0,img_shape - hole_w)
            offset_y = random.randint(0,img_shape - hole_h)
            mask[offset_y : offset_y+hole_h, offset_x : offset_x+hole_w] = 1.0
            mask = mask.unsqueeze(0)
            desire_msk_tensors.append(mask)
            
        
        else:    
            for name in range(len(r['class_ids'])):
                name_sig = class_names[r['class_ids'][name]]   
                #print("What is the shape of the r[mask]?")
                #print(mask_tensor[name].shape)
                sig_imgname.append(name_sig)
                count = sig_imgname.count(name_sig)
                instance_idx.append(count)
            print ("The instances in the single test image are as following: ")    
                
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                        class_names, instance_idx) 
            
            mask_tensor = trnsfm(r['masks'].astype(int))    # mask tensor for single image

           
                
                
        
            inputname = input("What type of instance do you want to elimilate ?  ")
            selected_idx = input("What is the index of that instance ")
            expand = input("Expand the mask by how many pixels? : ")
          
            aim_idx = [i for i, x in enumerate(sig_imgname) if x == inputname]
            print(aim_idx)
            for desired_msk in aim_idx:
               aim_msk_sigimg.append(mask_tensor[desired_msk])    # one mask on image is added to an array
            
            aim_msk_sigimg = torch.stack(aim_msk_sigimg)
            #print("What is the shape of stacked desired masks in one image????????????????????????????")
            #print(aim_msk_sigimg.shape)
            desire_msk_tensors.append(aim_msk_sigimg)                 # This is all desired masks in one batch
        
        
    
    output_masks = []
    #sig_mask, one image with all masks
    for sig_mask in range(len(desire_msk_tensors)):                # stack the mask of each image into a tensor
        selected_msk =segment_predict(desire_msk_tensors[sig_mask],selected_idx,expand)
        selected_msk=torch.from_numpy(selected_msk) # convert ndarray to tensor,
        #print("What is the shape of the mask? ", selected_msk.shape)
        #print("What is the type of selected_msk?",type(selected_msk))
        output_masks.append(selected_msk)
      
    stacked_msk_tensor = torch.stack(output_masks)   
    return stacked_msk_tensor    #msk tensors for different images in one batch

#---------------------------------------------------------------------------------------------------------
def find_LB_corner(right_bottom, left_top, global_size, local_size):
    """
    * inputs:
        - size (sequence, required)
                Size (w, h) of hole area.
        - mask_size (sequence, required)
                Size (w, h) of input mask.
    * returns:
            A sequence which is used for the input argument 'hole_area' of function 'gen_input_mask'.
    """
    
    if left_top[0].item() + local_size  < global_size:
        offset_x=left_top[0].item()
     
    elif right_bottom[0].item() < local_size:
         offset_x= int(0.5*(right_bottom[0].item()+left_top[0].item()-local_size)) 
    else:
        offset_x=right_bottom[0].item()-local_size
        
       
    
    if left_top[1].item() + local_size  < global_size:
        offset_y=global_size-(left_top.data[1].item()+local_size)
        
    elif right_bottom[1].item() < local_size:
        offset_y= int(global_size - 0.5*(right_bottom[1].item()+left_top[1].item()+local_size))
        
    else:    
        offset_y=global_size-right_bottom[1].item()
       
    return ((offset_x, offset_y), (local_size, local_size) )

def find_LB_corner_real(size, mask_size):
    """
    * inputs:
        - size (sequence, required)
                Size (w, h) of hole area.
        - mask_size (sequence, required)
                Size (w, h) of input mask.
    * returns:
            A sequence which is used for the input argument 'hole_area' of function 'gen_input_mask'.
    """
    mask_w, mask_h = mask_size
    harea_w, harea_h = size
    offset_x = random.randint(0, mask_w - harea_w)
    offset_y = random.randint(0, mask_h - harea_h)
    return ((offset_x, offset_y), (harea_w, harea_h))

def crop_real(x, area):
    """
    * inputs:
        - x (torch.Tensor, required)
                A pytorch 4D tensor (samples, c, h, w).
        - area (sequence, required)
                A sequence of length 2 ((x_min, y_min), (w, h)).
                sequence[0] is the left corner of the area to be cropped.
                sequence[1] is its width and height.
    * returns:
            A pytorch tensor cropped in the specified area.
    """
    xmin, ymin = area[0]
    w, h = area[1]
    
    return x[:, :, ymin : ymin + h, xmin : xmin + w]

def crop(x, area):
    """
    * inputs:
        - x (torch.Tensor, required)
                A pytorch 4D tensor (samples, c, h, w).
        - area (sequence, required)
                A sequence of length 2 ((x_min, y_min), (w, h)).
                sequence[0] is the left corner of the area to be cropped.
                sequence[1] is its width and height.
    * returns:
            A pytorch tensor cropped in the specified area.
    """
    crop_arr = []
    for item in range(x.shape[0]):
        
       
        x_sig = x[item]

     
        xmin, ymin = area[item][0]
        w, h = area[item][1]
       
       
        crop_arr.append(x_sig[ :, ymin : ymin + h, xmin : xmin + w])
         
    cropped_img = torch.stack(crop_arr)
    return cropped_img


def sample_random_batch(dataset, batch_size=32):
    """
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                An instance of torch.utils.data.Dataset.
        - batch_size (int, optional)
                Batch size.
    * returns:
            A mini-batch randomly sampled from the input dataset.
    """
    num_samples = len(dataset)
    batch = []
    for i in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        x = torch.unsqueeze(dataset[index], dim=0)
        batch.append(x)
    return torch.cat(batch, dim=0)


def poisson_blend(input, output, mask):
    """
    * inputs:
        - input (torch.Tensor, required)
                Input tensor of Completion Network.
        - output (torch.Tensor, required)
                Output tensor of Completion Network.
        - mask (torch.Tensor, required)
                Input mask tensor of Completion Network.
    * returns:
                Image tensor inpainted using poisson image editing method.
    """
    num_samples = input.shape[0]
    ret = []

    # convert torch array to numpy array followed by
    # converting 'channel first' format to 'channel last' format.
    input_np = np.transpose(np.copy(input.cpu().numpy()), axes=(0, 2, 3, 1))
    output_np = np.transpose(np.copy(output.cpu().numpy()), axes=(0, 2, 3, 1))
    mask_np = np.transpose(np.copy(mask.cpu().numpy()), axes=(0, 2, 3, 1))

    # apply poisson image editing method for each input/output image and mask.
    for i in range(num_samples):
        inpainted_np = blend(input_np[i], output_np[i], mask_np[i])
        inpainted = torch.from_numpy(np.transpose(inpainted_np, axes=(2, 0, 1)))
        inpainted = torch.unsqueeze(inpainted, dim=0)
        ret.append(inpainted)
    ret = torch.cat(ret, dim=0)
    return ret
