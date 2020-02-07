---
title: 'Object detection in computer vision'
date: 2020-02-07
permalink: /posts/2012/08/blog-post-4/
tags:
  - Computer Vision
  - Object Detection
---

Before 2013\. Traditional

Including VJ, HOG, DPM, SIFT, et al.

##### Two Stage

2013\. R-CNN

This is a milestone in object detection. In that work, proposals was firstly calculated using selective search. The process of selective search is shown below. 

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/selectivesearch.png" width="400" alt="selectivesearch">
<!-- ![AlexNet architecture](https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/alexnet.png)
 -->

After that, warping was used to normalize the size of proposals. Then, each normalized proposal was fed into AlexNet in order to get the proposal's category and location. Finally, non maximum suppression was used to remove the highly overlapped proposals. The framework of R-CNN is shown as below.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/rcnn.png" width="600" alt="rcnn">

2014\. SPPNet

In that work, the authors proposed spatial pyramid pooling layer (so called ROI Pooling) that can project the proposals of different sizes into a fixed length vector. In that case, the projection of SPP will be more precisely compared with warping.

In SPP layer, if we want to project a M\*N feature map into a 4\*4 feature map, the pooling size will be M/4 \* N/4. Pooling was calculated on each M/4 \* N/4 block, and finally a 4\*4 feature map will be obtained.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/rcnn.png" width="400" alt="spp1">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/rcnn.png" width="600" alt="spp2">

2015\. Fast-RCNN

In that work, the authors make a little change on the process and the feature map of whole image only need to be calculated once. 

In reference, the image was fed into selective search algorithm and VGG backbone simultaneously. Then, the proposals from the raw image were projected into the feature map, and ROI Pooling was used to project the feature maps of proposals into a fixed size. Finally, two fully connected layers were used to extract the feature vectors from proposals, and one fully connected layer plus a softmax layer or a regression layer were used to classify and localize the proposals.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/fastrcnn.png" width="600" alt="fastrcnn">




