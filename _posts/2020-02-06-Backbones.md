---
title: 'Backbones in computer vision'
date: 2020-02-06
permalink: /posts/2020/02/blog-post-1/
tags:
  - Computer Vision
  - Backbone
---

**This is a brief description of well known backbones, which can help people remember the knowledge.**

**2012\. AlexNet**

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/alexnet.png" width="600" alt="AlexNet architecture">
<!-- ![AlexNet architecture](https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/alexnet.png)
 -->

**1990\. RNN**

The work used history memory and current input to calculate current output.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/rnn.png" width="300" alt="RNN architecture">

**1997\. LSTM**

On the basis of RNN, LSTM introduced several gates that based on sigmoid function to control the information transform.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/lstm.png" width="400" alt="LSTM architecture">
<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/lstm1.png" width="300" alt="LSTM architecture">

The following one is forget gate, it controls the amount of passed history memory λCt-1.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/lstm2.png" width="400" alt="LSTM architecture">

The following one is input gate, it demetermines the current state input information Pt.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/lstm3.png" width="400" alt="LSTM architecture">

The following one is memory gate, the passed history memory λCt-1 plus the current state input information Pt formed the current memory Ct.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/lstm4.png" width="400" alt="LSTM architecture">

The following one is the unit output Ht, it was formed based on the passed activation (tanh) of current memory Ct.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/lstm5.png" width="400" alt="LSTM architecture">

**2015\. VGG**

The work, small convolutional sizes such as 3\*3 and 5\*5 were used to learn high level features. The work used a deep network to improve network's learning ability.

This network has too many parameters (138M).

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/vgg.png" width="600" alt="VGG16 Architecture">

**2015\. InceptionV1 (GoogLeNet)**

In this work, convolutional layers were used to extract feature maps. Therefore, the number of parameters is obviously reduced compare with previous architectures. Moreover, this work proposed Inception module that can parallelly calulate multi-scale features, and auxiliary classifiers were used to increase the gradient signal and provide additional regularization.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/inceptionv12.png" width="400" alt="InceptionV1 block">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/inceptionv11.png" width="600" alt="InceptionV1 Architecture">

**2015\. InceptionV2**

The work, the authors firstly point out gradient vanishing and explorsion problems and Internal Covariate Shift. Then, they used batch normalization to solve these problems and obtained good results.

The process of batch normalization is shown as below. The distribution of weight of each layer was firstly changed to standard normal distribution using the each layer's weight of all samples in each batch, then γ and β were used to stretch the normalized distribution in order to keep the nonlinear learning ability. In convolutional layers, each feature map has one mean and variance, which means a feature map (N sample, C channel, W\*H size) will have C mean values and C variance values.

In the reference step, mean and variance were calculated using all samples in the training set.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/inceptionv2.png" width="400" alt="Batch normalization">

**2015\. InceptionV3**

The work, the authors introduced asymmetric convoltions, which means a 3\*3 filter could be replaced by one 3\*1 filter and one 1\*3 filter.

**2016\. InceptionV4 (Inception-ResNet)**

The work, the authors introduced Inception into ResNet architecture as shown below.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/inceptionv4.png" width="200" alt="InceptionV4 Architecture">

**2016\. ResNet**

The work used a deep network with residual bottlenecks to solve the gradient vanishing and explorsion problem and network degradation issue.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/resnet2.png" width="200" alt="ResNet Block">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/resnet1.png" width="500" alt="ResNet Block at Pooling">

**2016\. Identity mapping**

Based on the work of resnet, the work achieved indentity mapping using a new block, which means signal can be transformed between any layer.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/identitymapping1.png" width="300" alt="ResNet with identity mapping">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/identitymapping2.png" width="400" alt="Differentiation of identity mapping">

**2017\. DenseNet**

The work introduced dense block. In a dense block, the later layers have identity mapping with all the previous layer, and the multi-layer's features were combined by concatenating them.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/densenet.png" width="600" alt="DenseNet architecture">

**2017\. ResNeXt**

The work proposed an extra dimention called cardinality, which means the block could be constructed based on multiple parallel bottlenecks. This structure could improve model's expressive ability. The author used **Group Convolution** to achieve this idea, and the best parameters are width 4 and cardinality 32.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/groupconvolution.png" width="300" alt="Group Convolution">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/resnext.png" width="500" alt="ResNeXt architecture">

**2017\. Xception**

The work used **Depthwise Separable Convolution** to construct the model, and it could achieved the state-of-the-art accuracy with fewer parameters.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/dsc.png" width="300" alt="Depthwise Separable Convolution">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/xception.png" width="600" alt="xception architecture">

**2017\. SENet**

SENet won the last ILSVRC 2017 challenge.
The idea is to calculate the concolution based on each channel's weight. In that method, the normal convolution is firstly used to calculate the next feature map M. Then, the global average pooling is used on M and form a 1\*1\*C vector. After that, two fully connected layers are used to calculate the weight of each channel and form a 1\*1\*C vector W. Finally, each channel in M will multiply corresponding weight and form the ajusted feature map.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/se.png" width="300" alt="SENet architecture">
<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/se1.png" width="200" alt="SENet architecture">

**2017\. NAS**

The work is a millstone, which used reinforcement to search the optimal neural network architecture.

The work is based on reinforment learning and RNN to form a unfixed length description of network architecture, such as filter number, filter size, activation, skip connection, batch normalization, etc. The evaluation criteria is based on the validation accuracy.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/nas.png" width="300" alt="NAS">

During training, the agency (RNN or LSTM) produces 100 replicas, and each replica produces 8 architecture samples. After 50 ephochs's training, the performance of samples feedbacks to the agency and updates its parameters. The initial depth of the architecture is 6, the depth will plus 2 after sampling 1600 architecture samples. After sampling 12800 architecture samples, the architecture will be fixed, and grid searching will be used to find the optimal learning rate, weight decay, batch parameter, etc.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/nas1.png" width="500" alt="NAS">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/nas2.png" width="300" alt="NAS architecture">

Currently, the neural architecture search mainly contains six methods: 1) searching the whole architecture, and after reaching a certain iteration, the network could be deeper; 2) searching the optimal block, and contruct the network by stacking blocks together, such as MnasNet; 3) Contructing a set of architectures and search the optimal network in the set; 4) using graph search method to find the best architecture; 5) using evolutionary algorithm to search the optimal architecture, which means architures will produced and after competition, crossover and variation the architure will be better; 6) projecting the descrete parameters to a continuous space, and use gradient decent to find the optimal solution instead of using polity gradient.

**2019\. EfficientNet**

The work is a state-of-the-art backbone that outperforms other methods.

Ths work is based on several observations: 1) higher input resolution and more channel will bring more fine-grained features; 2) deeper network will bring richer and complex features; 3) the representation of model will improved with the increase of resolution, width (channel) and depth, however, the improvement will reach saturation; 4) the three factors influence each other.


Based on these observations, the work firstly find a optimal backbone based on MnasNet, which consists repeated MobileNetV2 and SE blocks. 

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/efficientnet2.png" width="400" alt="EfficientNet">

Then, the work focuses on how to rescale the architecture in order to obtain a better representation. The authors bind the multiplication of the square of resolution, width and depth to 2. After that, the author search the best paramters under the contraints. Finally, the author change Φ to pursue a better representation.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/efficientnet.png" width="300" alt="EfficientNet">
<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/efficientnet1.png" width="200" alt="EfficientNet">
