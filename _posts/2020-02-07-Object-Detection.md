---
title: 'Object detection in computer vision'
date: 2020-02-07
permalink: /posts/2020/02/blog-post-02/
tags:
  - Computer Vision
  - Object Detection
---

Before 2013\. Traditional

Including VJ, HOG, DPM, SIFT, et al.

#### General Parameters

Scale jittering, rotation for data augmentation;

Pre-training using Imagenet;

SGD, Momuntum is 0.9 with a weight decay 0.0005;

0.001 initial learning rate, and devide by 10 after N0K batches;

Focal loss for one-stage methods;

RPN for two-stage methods;

Image centric sampling with 2 images and 256 proposals;

IoU > 0.5 is positive sample;

Mini-batch size is 256 and positive:negative=1:3;

Non maximum suppression (NMS) for removing highly overlapped proposals;




### Two Stage

2013\. R-CNN

**Highlight: Selective search && Pre-training**

This is a milestone in object detection. In that work, proposals was firstly been calculated using selective search. The process of selective search is shown below. 

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/selectivesearch.png" width="400" alt="selectivesearch">
<!-- ![AlexNet architecture](https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/alexnet.png)
-->

After that, warping was used to normalize the size of proposals. Then, each normalized proposal was fed into AlexNet in order to get the proposal's category and location. Finally, non maximum suppression was used to remove the highly overlapped proposals. The framework of R-CNN is shown as below.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/rcnn.png" width="600" alt="rcnn">

2014\. SPPNet

**Highlight: SPP layer (ROI Pooling)**

Based on the R-CNN architecture, the authors proposed spatial pyramid pooling layer (so called ROI Pooling) that can project the proposals of different sizes into a fixed length vector. In that case, the projection of SPP will be more precisely compared with warping.

In SPP layer, if we want to project a M\*N feature map into a 4\*4 feature map, the pooling size will be M/4 \* N/4. Pooling was calculated on each M/4 \* N/4 block, and finally a 4\*4 feature map will be obtained.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/spp1.png" width="400" alt="spp1">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/spp2.png" width="600" alt="spp2">

2015\. Fast R-CNN

**Highlight: Multi-task loss**

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/fastrcnn.png" width="300" alt="fastrcnn1">

In that work, the authors make a little change on the process and the feature map of whole image only need to be calculated once. 

In reference, the image was fed into selective search algorithm and VGG backbone simultaneously. Then, the proposals from the raw image were projected into the feature map, and ROI Pooling was used to project the feature maps of proposals into a fixed size. Finally, two fully connected layers were used to extract the feature vectors from proposals, and one fully connected layer plus a softmax layer or a regression layer were used to classify and localize the proposals.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/fastrcnn.png" width="600" alt="fastrcnn">

2015\. Faster R-CNN

**Highlight: Region proposal network**

In that work, the authors used region proposal network (RPN) to replace the selective search for proposal mining, and realized an end-to-end network. 

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/fasterrcnn.png" width="400" alt="fasterrcnn">

In inference, the image firstly fed into VGG to get the feature map. Then, proposals were calculated by feeding feature map into RPN network. After that, each proposal was resized into a fixed size using ROI Pooling, and fully connected layers were used for classification and localization.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/fasterrcnn1.png" width="400" alt="fasterrcnn1">

2016\. OHEM

**Highlight: Hard negative mining**

This work proposed a method for hard negative mining issue. In this work, proposals were firstly calculated using RPN and fed into detection network A. Then, sorted by loss, top-N proposals were selected and fed into detection network B for training. A and B have same architecture and parameters, and the up-to-date parameters of A will be copied to B. In that way, the network will more focus on hard problems.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/ohem.png" width="600" alt="ohem">
