---
title: 'Backbones in computer vision'
date: 2020-02-06
permalink: /posts/2020/02/blog-post-1/
tags:
  - Computer Vision
  - Backbone
---

2012\. AlexNet

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/alexnet.png" width="600" alt="AlexNet architecture">
<!-- ![AlexNet architecture](https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/alexnet.png)
 -->

2015\. InceptionV1 (GoogLeNet)

In this work, convolutional layers were used to extract feature maps. Therefore, the number of parameters is obviously reduced compare with previous architectures. Moreover, this work proposed Inception module that can parallelly calulate multi-scale features, and auxiliary classifiers were used to increase the gradient signal and provide additional regularization.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/inceptionv12.png" width="400" alt="InceptionV1 block">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/inceptionv11.png" width="600" alt="InceptionV1 Architecture">

2015\. InceptionV2

In this work, the authors firstly point out gradient vanishing and explorsion problems and Internal Covariate Shift. Then, they used batch normalization to solve these problems and obtained good results.

The process of batch normalization is shown as below. The distribution of weight of each layer was firstly changed to standard normal distribution using the each layer's weight of all samples in each batch, then γ and β were used to stretch the normalized distribution in order to keep the nonlinear learning ability. In convolutional layers, each feature map has one mean and variance, which means a feature map (N sample, C channel, W\*H size) will have C mean values and C variance values.

In the reference step, mean and variance were calculated using all samples in the training set.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/inceptionv2.png" width="400" alt="Batch normalization">

2015\. InceptionV3

In this work, the authors introduced asymmetric convoltions, which means a 3\*3 filter could be replaced by one 3\*1 filter and one 1\*3 filter.

2016\. InceptionV4

In this work, the authors introduced Inception into ResNet architecture as shown below.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/inceptionv4.png" width="300" alt="InceptionV4 Architecture">

2015\. VGG

In this work, small convolutional sizes such as 3\*3 and 5\*5 were used to learn high level features. This work used a deep network to improve network's learning ability.

This network has too many parameters (138M).

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/vgg.png" width="600" alt="VGG16 Architecture">

2016\. ResNet

This work used a deep network with residual bottlenecks to solve the gradient vanishing and explorsion problem and network degradation issue.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/resnet2.png" width="400" alt="ResNet Block">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/resnet1.png" width="400" alt="ResNet Block at Pooling">

2016\. Identity mapping

Based on the work of resnet, this work achieved indentity mapping using a new block, which means signal can be transformed between any layer.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/identitymapping1.png" width="400" alt="ResNet with identity mapping">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/identitymapping2.png" width="400" alt="Differentiation of identity mapping">

2017\. DenseNet

This work introduced dense block. In a dense block, the later layers have identity mapping with all the previous layer, and the multi-layer's features were combined by concatenating them.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/densenet.png" width="600" alt="DenseNet architecture">


