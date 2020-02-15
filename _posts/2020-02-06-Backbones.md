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

On the basis of RNN, LSTM introduced several **gates** that based on sigmoid function to control the information flow.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/lstm.png" width="400" alt="LSTM architecture">
<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/lstm1.png" width="300" alt="LSTM architecture">

The following one is the forget gate, which controls the amount of passed history memory λCt-1.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/lstm2.png" width="400" alt="LSTM architecture">

The following one is the input gate, which demetermines the current state input information Pt.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/lstm3.png" width="400" alt="LSTM architecture">

The following one is the memory gate, the passed history memory λCt-1 plus the current state input information Pt will form the current memory Ct.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/lstm4.png" width="400" alt="LSTM architecture">

The following one is the unit output Ht, it is base on the passed activation (tanh) of current memory Ct.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/lstm5.png" width="400" alt="LSTM architecture">

**2015\. VGG**

Small convolutional sizes such as 3\*3 and 5\*5 were used in the work to learn high level features, a **deep** network was to improve network's learning ability.

This network has too many parameters (138M).

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/vgg.png" width="600" alt="VGG16 Architecture">

**2015\. InceptionV1 (GoogLeNet)**

**Only convolutional layers** were used in the work to extract feature maps. Therefore, the number of parameters is obviously reduced compared with previous work. Moreover, the work proposed **Inception module** that can parallelly calulate multi-scale features, and **auxiliary classifiers** were used to increase the gradient signal and provide additional regularization.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/inceptionv12.png" width="400" alt="InceptionV1 block">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/inceptionv11.png" width="600" alt="InceptionV1 Architecture">

**2015\. InceptionV2**

The authors firstly point out gradient vanishing and explorsion problems and Internal Covariate Shift. Then, they used **batch normalization** to solve these problems and the batch normalization could obtain good results.

The process of batch normalization is shown as below. The distribution of weight of each layer was firstly changed to standard normal distribution using the corresponding layer's weights of all samples in each batch, then γ and β were used to stretch the normalized distribution in order to keep the nonlinear learning ability. In convolutional layers, each channel has one mean and variance, which means a feature map (N sample, C channel, W\*H size) will have C mean values and C variance values.

In the reference step, mean and variance were calculated using all samples in the training set.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/inceptionv2.png" width="400" alt="Batch normalization">

**2015\. InceptionV3**

The authors introduced **asymmetric convoltions**, which means a 3\*3 filter could be replaced by one 3\*1 filter and one 1\*3 filter.

**2016\. InceptionV4 (Inception-ResNet)**

The authors introduced Inception modules into ResNet architecture as shown below.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/inceptionv4.png" width="200" alt="InceptionV4 Architecture">

**2016\. ResNet**

The work used a **deep network with residual bottlenecks** to solve the gradient vanishing and explorsion problem and network degradation issue.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/resnet2.png" width="200" alt="ResNet Block">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/resnet1.png" width="500" alt="ResNet Block at Pooling">

**2016\. Identity mapping**

Based on the work of resnet, the work achieved **indentity mapping** using a new block, which means signal can be transformed between any layer straightforwardly.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/identitymapping1.png" width="300" alt="ResNet with identity mapping">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/identitymapping2.png" width="400" alt="Differentiation of identity mapping">

**2017\. DenseNet**

The work introduced dense block. In a dense block, the **later layers have identity mapping with all the previous layers**, and the multi-layer's features were concatenated with each other.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/densenet.png" width="700" alt="DenseNet architecture">

**2017\. ResNeXt**

The work proposed an extra dimention called **cardinality**, which means the block could be constructed based on multiple parallel bottlenecks. This structure could improve model's expressive ability. The authors used **Group Convolution** to achieve this idea, and the best parameters are width 4 and cardinality 32.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/groupconvolution.png" width="400" alt="Group Convolution">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/resnext.png" width="600" alt="ResNeXt architecture">

**2017\. Xception**

The work used **Depthwise Separable Convolution** to construct the model, and it could achieved the state-of-the-art accuracy with fewer parameters.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/dsc.png" width="300" alt="Depthwise Separable Convolution">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/xception.png" width="700" alt="xception architecture">

**2017\. SENet**

SENet won the last ILSVRC 2017 challenge. The idea of the work is that different channels are not important equally. In that method, the normal convolution was firstly used to calculate the feature map M. Then, the global average pooling was used on M and form a 1\*1\*C vector. After that, two fully connected layers were used to calculate the weight of each channel and form a 1\*1\*C vector W. Finally, each channel in M multiplied by corresponding weight and form the ajusted feature map.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/se.png" width="400" alt="SENet architecture">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/se1.png" width="300" alt="SENet architecture">

**2017\. NAS**

The work is a millstone, which used reinforcement to search the optimal neural network architecture.

The work was based on **reinforment learning and RNN** to form an **unfixed length description of network** architecture, that including filter number, filter size, activation, skip connection, batch normalization, etc. The evaluation criteria of agency was based on the validation accuracy.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/nas.png" width="300" alt="NAS">

During training, the agency (RNN or LSTM) produced 100 replicas, and each replica produced 8 architecture samples. After 50 ephochs's training, the representation of samples were fed into the agency and updatd its parameters. 

The initial depth of the architecture was set to 6, the depth was plus 2 after sampling 1600 architecture samples. After sampling 12800 architecture samples, the architecture was fixed, and grid searching was used to find the optimal learning rate, weight decay, batch parameter, etc.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/nas1.png" width="600" alt="NAS">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/nas2.png" width="300" alt="NAS architecture">

Currently, the neural architecture search mainly contains six methods: 1) searching the whole architecture, and after reaching a certain iteration, the network could be deeper, such as NAS; 2) searching the optimal block, and contruct the network by stacking blocks together, such as MnasNet; 3) Contructing a set of architectures and search the optimal network in the set; 4) using graph search methods to find the best architecture; 5) using evolutionary algorithms to search the optimal architecture, which means architures will be produced firstly, and after competition, crossover and variation the left architures will be better; 6) projecting the descrete parameters to a continuous space, and use gradient decent to find the optimal solution instead of using polity gradient.

**2019\. EfficientNet**

The work is a state-of-the-art backbone that outperforms other methods. Its idea is based on several observations: 1) higher input resolution and more channels will bring more fine-grained features; 2) deeper network will bring richer and complex features; 3) the representation of model will be improved with the increase of resolution, width (channel) and depth, however, the improvement will reach saturation; 4) the three factors could influence each other.

Based on these observations, the work firstly found a optimal backbone based on MnasNet, which consists of repeated MobileNetV2 and SE blocks. 

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/efficientnet2.png" width="400" alt="EfficientNet">

Then, the work focused on how to rescale the architecture in order to obtain a better representation. The authors restricted the multiplication of the square of resolution, width and depth to 2. After that, the authors searched the best parameters under the contraints and fixed them. Finally, the authors changed Φ to pursue a better representation.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/efficientnet.png" width="300" alt="EfficientNet">
<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/efficientnet1.png" width="200" alt="EfficientNet">

**2017\. MobileNet V1**

The network architecture is similiar to Xception, the difference is that MobileNet V1 does not contain skip connection and the **depthwise sperable convolution** in MobileNet V1 has two activations. MobileNet V1 is also **compressible** since it has parameters α and β to control the channel number and input image size, respectively.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/mobilenetv1.png" width="200" alt="MobileNet V1">

**2018\. MobileNet V2**

The blocks (**inverted residuals**) of the work is somehow like ResNeXt and ShuffleNet V1. MobileNet V2 contains skip connection, and it used the first 1\*1 convolution (expansion layer) to upgrade the channel number in order to prevent information lost caused by activation. Moreover, the authors used **ReLU6** in the work because of its robustness in low-precision computation.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/mobilenetv2.png" width="200" alt="MobileNet V2">

**2019\. MobileNet V3**

Like EfficientNet, the work used **MnasNet** as the initial backbone. Then, **NetAdapt** was used to optimze the initial backbone based on the goal of latency. NetAdapt firstly proposed several samples based on the contraint that new samples should have less lantency compared with initial samples, and these sample networks were then fine-tuned for several epochs. Based on the evaluation function argmax[ΔAcc/ΔLatency], the best model was chosen. Finally, the model was trained from scratch. The work also used **Hswish** as the activation function to pursue a better representation.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/mobilenetv3.png" width="300" alt="MobileNet V3">

**2017\. ShuffleNet V1**

The authors firstly used **1\*1 group convolution** to reduce the dense calculation. They also proposed **channel shuffle** to realize information flow between different groups. Moreover, they found that the benefits of channel shuffle increase with the increase of groups under fixed width.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/shufflenetv1.png" width="400" alt="channel shuffule">

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/shufflenetv11.png" width="400" alt="ShuffleNet V1 Block">

**2018\. ShuffleNet V2**

The work firstly pointed out that the traditional FLOPs evaluation method is not enough for model evaluation because it does not consider the memory access cost (MAC) and platform. In that case, evaluation such as inference time should be used to evaluate the models.

By analyzing the cost of resource, the authors found that: 1) balanced pointwise convolution channel (1:1) between input and output could reduce MAC; 2) too much group convolution will increase MAC; 3) too much fragments (groups) will reduce parallelism; 4) element-wise operation will cost time, such as shortcut connection and ReLU. Based on these observations, the author proposed a new block.
1) Each unit will cut the input channel into two equal parts, one is used for shortcut connection, and the other is used for convolution. Therefore, the outputs will concatenate with each other and it could obey the 4th rule.
2) 1\*1 group convolution is no longer used to obey 2th rule.
3) channel numbers of input and output are equal in each unit to obey 1th rule.
4) the output of bottleneck and skip connection are concatenated with each other to obey 1th rule. Moreoever, channel shuffle is used to make sure information flow.
5) during downsampling, the channels will not split into two parts. Therefore, the channel number will double in the output, it could obey 4th rule.

<img src="https://raw.githubusercontent.com/Robert-BoMiao/Robert-BoMiao.github.io/master/images/blog_images/shufflenetv2.png" width="400" alt="ShuffleNet V2">
