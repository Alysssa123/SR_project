# SR_project
The project aims to generate super-resolution imagery from a given low-resolution image with the model optimized based on SRGAN. It could achieve great results with the recovery of natural textural detail.

# Contents of the SR_project
1. Scripts: The improved SR model containing data processing, training, testing and evaluation functions.
2. Bicubic_SR: Run the script bicubicSR.py to generate SR images with bicubic interpolation method.
3. Demo_SR.ipynb: It shows entire running proccess of the SR code and involves a short Youtube video.
4. Eval_SR.ipynb: It shows the comparisons of quantitive results of SR models.

# Links

Youtube video link: https://youtu.be/EbSQVd0JNPQ

Github link: https://github.com/Alysssa123/SR_project.git

# Tools of the project
Python >= 3.6

PyTorch >= 1.0

NVIDIA GPU + CUDA

# Dataset download
There are three kinds of datasets: training dataset, validation dataset, and testing dataset. Download the GT images from below link:
1. Training:
      
DIV2K: 800 https://data.vision.ee.ethz.ch/cvl/DIV2K/ DIV2K_train_LR_bicubic_X4
      
Flick2k: 2650 https://drive.google.com/drive/folders/15Tj0Hke4xQxF5ahs2-bnK9lCe8il0RQm?usp=sharing

2. Validation:
      
Div2k: 200 https://data.vision.ee.ethz.ch/cvl/DIV2K/ DIV2K_valid_LR_bicubic_X4
  
3. Testing:
      
Flick-Faces-HQ(FFHQ): 1000 https://drive.google.com/drive/folders/15Tj0Hke4xQxF5ahs2-bnK9lCe8il0RQm?usp=sharing
      
Self-collected dataset: 126 https://drive.google.com/drive/folders/14JMoUQhplMTBPW_A2knzKdIEvi8I8rQD?usp=sharing
 
 # Data processing
 We need do the following pre-processing before training. 
 
 Step 1: Data augmentation
    
Enrich our training data by appling transformations of vertical and horizontal flip and rotations with 90, 180 and 270 angles. One image could be augmented to six images by running the script augment_imgs.py. All images in DIV2k and Flick2K use the same augmentation.
 
 Step 2: extract sub images
    
DIV2K has 2 K resolution but the training patches are 128 × 128. We crop one 2K resolution image into 40 480 × 480 sub-images by runing the script extract_subimgs_single.py.
 
 Step 3: Down sample
 
 Run the script GEN_LR.py to Generate low-resolution counterparts. Make sure the LR and GT pairs have the same name.
 
 Step 4: create train lmdb
 
 Create LMDB files with training data and validation data for greatly speeding up IO and CPU decompression during training.
 
# Train
Modify the configuration file Train/train_SRGAN.json

Run command: python Train/train.py -opt Train/train_SRGAN.json

# Test
Firstly download the testing datasts from above link and generate LR counterparts. 

Run command: Python Test/test.py models/SR_result.pth

# Evaluation
Evaluate the model with PSNR and SSIM metric on the test dataset

Run the script python calculate_PSNR_SSIM.py 


 
 
 
 
 
 
 
