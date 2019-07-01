# FYP-Image Completion Using MASK_RCNN and GLCIC
## Introduction
This repository includes the code for my final year project, using Mask-RCNN [Jiaming He et. al. 2018] and GLCIC [ Iizuka et. al. 2017] to remove unwanted objects from image with automatic mask proposal. It is highly recommended to run this project in colab environment with GPU notbook becasue most of dependencies are pre-installed in colab. Alternatively, spyder or command line are all possible options. 

**To run the code on colab, the project needs first be downloaded to the user's google drive, then by running the first cell in the notebook, the notebook can be connected to the google drive**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hFF1okczZxFA7QFUXbSJENjThbcJuSHB?authuser=4)

**Part of the code is borrowed from otenim's implementation of GICIC and matterport's implementation of Mask-RCNN, great aprreciation to these works.**

![Demo](https://raw.githubusercontent.com/zw4315/FYP/master/results/result/demo.jpg)

## Dependencies
If the code is run in colab, only **pyamg** and **tensorflow-gpu** need installing, and they are written in the notebook so please just follow the order to run each cell. However, if the code is run in local python 3.6.5 environment, further dependencies need installing as well including:
* torch: 1.0.1post2
* torchvision: 0.2.2.post3
* tqdm: 4.31.1
* Pillow: 5.4.1
* numpy: 1.16.2
* scipy: 1.2.1
* GPU: Nvidea Geforce GPU with cuda, 1050Ti at least 


