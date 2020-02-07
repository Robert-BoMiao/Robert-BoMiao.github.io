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

# General Parameters

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




# Two Stage

**2013\. R-CNN**

**Highlight: Selective search && Pre-training**

This is a milestone in object detection. In that work, proposals was firstly been calculated using selective search. The process of selective search is shown below. 

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/selectivesearch.png" width="400" alt="selectivesearch">
<!-- ![AlexNet architecture](https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/alexnet.png)
-->

After that, warping was used to normalize the size of proposals. Then, each normalized proposal was fed into AlexNet in order to get the proposal's category and location. Finally, non maximum suppression was used to remove the highly overlapped proposals. The framework of R-CNN is shown as below.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/rcnn.png" width="600" alt="rcnn">

**2014\. SPPNet**

**Highlight: SPP layer (ROI Pooling)**

Based on the R-CNN architecture, the authors proposed spatial pyramid pooling layer (so called ROI Pooling) that can project the proposals of different sizes into a fixed length vector. In that case, the projection of SPP will be more precisely compared with warping.

In SPP layer, if we want to project a M\*N feature map into a 4\*4 feature map, the pooling size will be M/4 \* N/4. Pooling was calculated on each M/4 \* N/4 block, and finally a 4\*4 feature map will be obtained.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/spp1.png" width="400" alt="spp1">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/spp2.png" width="600" alt="spp2">

**2015\. Fast R-CNN**

**Highlight: Multi-task loss**

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/fastrcnn1.png" width="300" alt="fastrcnn1">

In that work, the authors make a little change on the process and the feature map of whole image only need to be calculated once. 

In reference, the image was fed into selective search algorithm and VGG backbone simultaneously. Then, the proposals from the raw image were projected into the feature map, and ROI Pooling was used to project the feature maps of proposals into a fixed size. Finally, two fully connected layers were used to extract the feature vectors from proposals, and one fully connected layer plus a softmax layer or a regression layer were used to classify and localize the proposals.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/fastrcnn.png" width="600" alt="fastrcnn">

**2015\. Faster R-CNN**

**Highlight: Region proposal network**

In that work, the authors used region proposal network (RPN) to replace the selective search for proposal mining, and realized an end-to-end network. 

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/fasterrcnn.png" width="400" alt="fasterrcnn">

In inference, the image firstly fed into VGG to get the feature map. Then, proposals were calculated by feeding feature map into RPN network. After that, each proposal was resized into a fixed size using ROI Pooling, and fully connected layers were used for classification and localization.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/fasterrcnn1.png" width="400" alt="fasterrcnn1">

**2016\. OHEM**

**Highlight: Hard negative mining**

This work proposed a method for hard negative mining issue. In this work, proposals were firstly calculated using RPN and fed into detection network A. Then, sorted by loss, top-N proposals were selected and fed into detection network B for training. A and B have same architecture and parameters, and the up-to-date parameters of A will be copied to B. In that way, the network will more focus on hard problems.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/ohem.png" width="600" alt="ohem">

**2017\. R-FCN**

**Highlight: Proposal sensitive score map**

In that work, the authors proposed a totally convolutional network. After feature map calculation, proposed proposal sensitive score map was used to solve the redundant calculation issue in previous methods and thus improve the inference speed. 

The proposal sensitive score map has k\*k\*(C+1) channels. In that work, the author define k=9, which means the proposal could be divided into 9 parts (top-left, top-right, etc.). After proposals (calculated by RPN) project into the score map, the proposals will be divided into k identical parts, and the top-left parts will select the corrosponding C+1 channel features at the top-left and change them into a 1\*1\*(C+1) feature map using pooling. Then, a k\*k\*(C+1) feature map will be obtained, and after another pooling, the C+1 dimentional feature vector will be fed into softmax and calculate the results.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/rfcn1.png" width="600" alt="rfcn">

**2017\. Light Head R-CNN**

**Highlight: Proposal sensitive score map with k\*k\*10 channels**

In that work, the authors proposed a proposal sensitive score map like R-FCN. However, the proposed score map only contain k\*k\*10 channels. Therefore, less memory will be needed compared with R-FCN (k\*k\*(C+1)). Finally a fully connected layer was used to extract high-level feature.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/lightheadrcnn.png" width="600" alt="lightheadrcnn">

**2017\. FPN**

**Highlight: Feature Pyramid ---> Multi-scale-level proposals**

In that work, the authors proposed a feature pyramid network. That method is based on the knowledge that the top layer has more high-level semantic information and the bottom layer haslow-level higher resolution. 

During inference, feature maps with different sizes were firstly calculated, then the bottom layer will merge the information of higher layers. Finally, RPN and detection head of other methods could be used for detection.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/fpn.png" width="600" alt="fpn">

**2017\. Mask R-CNN**

**Highlight: ROI Align && Mask prediction && Class+BBox+Mask Loss**

In that work, the author proposed ROI Alignment to solve the misalignment issue in ROI Pooling. ROI alignment is based on bilinear interpolation. The process of ROI alignment is shown as below.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/maskrcnn.png" width="400" alt="maskrcnn">

Moreover, the network is also used for instance segmentation problem, the results shown that optimize the loss of mask prediction can also improve the detection accuracy.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/maskrcnn1.png" width="600" alt="maskrcnn1">

**2018\. Cascade R-CNN**

**Highlight: Cascade block**

That work mainly focus on the optimization of detection head. The authors found that the detection performance tends to degrade with the increase of threshold. That phenomenon maybe caused by the different thresholds between traning and test. In that case, cascade block can solve this problem. 

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/cascadercnn1.png" width="600" alt="cascadercnn">

After each prediction the model will be more confident to the positive samples. Therefore, cascade block with different threshold will have similiar sample distribution because the 2nd stage with higher threshold will receive samples with higher confidence. During reference, the model is thus more robust to different thresholds.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/cascadercnn.png" width="600" alt="cascadercnn">


# One Stage

**2016\. YOLO V1**

**Highlight: Very fast, end to end**

In that work, GoogLeNet plus 5 extra convolutional layes were used to extract features, and 7\*7\*30 feature map was obtained. The 7\*7\*30 feature map means the image could be divided into 7\*7 parts, each part may contain 2 objects (2 confidence + 8 bbox + 20 classes).

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/yolov1.png" width="600" alt="yolov1">

**2016\. YOLO V2**

**Highlight: K-means for the calculation of Default anchors, Multi-scale feature concatenation, WordTree classification**

The final 7\*7\*125 feature map represents 7\*7 parts, each part may contain 5 anchors, each anchor has 1 confidence, 4 positions and 20 classes.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/yolov2.png" width="600" alt="yolov2">

**2018\. YOLO V3**

**Highlight: DarkNet + FPN**

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/yolov3.png" width="600" alt="yolov3">

**2016\. SSD**

**Highlight: Multi-scale feature maps, Default anchors**

In that work, VGG and extra convolutional layers were used to calculate feature maps at different resolution. Then, the chosen 6 layers were used for detection.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/ssd.png" width="600" alt="ssd">

**2016\. RetinaNet**

**Highlight: Focal Loss**

The authors found that its complex to select hard samples based on mannual rules in training and the huge number of easy negative samples will exert an adverse impact on the learning. In that case, focal loss could help model by giving more weights to the hard samples and the initialization of bias can solve the problem of different ratio between positive and negative samples.

The folcal loss is calculated as below:

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/focalloss.png" width="300" alt="focalloss">

α could decide the importance of positive samples, and γ could decide the decay speed of the weight with the increase of confidence.

To solve the sample imbalance issue during training, the weights is initialized to zero mean, and bias is initialized to -log((1-π)/π)), where π=0.01. In that case, after the calculation of sigmoid, the activation will be π. Therefore, the loss of positive samples will be -log(π)=2 and the loss of negative samples will be -log(1-π)≈0.004. Finally, positive samples will have more weight at the begining of training.

**2017\. Soft NMS**

**Highlight: Soft NMS**

Instead of rudely removing highly overlapped proposals, Soft NMS choose the proposals with highest score and reduce the confidence of other overlapped proposals at each iteration.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/softnms.png" width="300" alt="softnms">

**2018\. RFB Net**

**Highlight: Inception block, dialated convolution and pooling**

This work is based on the visual principle of Biology. After Combining Inception block with dilated convolution, the recptive field that obey the visual principle was obtained. Then, detection was finished using multi-scale feature maps.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/rfbnet.png" width="300" alt="rfbnet">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/rfbnet1.png" width="300" alt="rfbnet1">

**2019\. M2Det**

**Highlight: FFM, TUM, SFAM**

The authors aims to let the model learn both multi-level and multi-scale features. In this method, images was firstly fed into a backbone, and two of all feature maps will concatenate into one feature map I. Then I was fed into TUM that like a feature pyramid network. After that, the low-level multi-scale feature maps were obtained. Then, the feature map O with highest resolution was choosen from the low-level multi-scale feature maps. I and O were then concatenated into one feature map through FFM2 and fed into the next TUM. Finally, low&mid&high-level multi-scale feature maps could be obtained, feature maps with same scale will concatenate with each other, and the feature maps were reweighted using the mean value of each channel.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/m2det.png" width="300" alt="m2det">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/object_detection/m2det1.png" width="300" alt="m2det1">

