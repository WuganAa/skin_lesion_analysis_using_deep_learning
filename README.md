# Using deep learning network for skin lesion analysis of melanoma detection

This is an experimental report of [Skin Lesion Analysis Using Melanoma Detection Based on Deep Learning](http://xuganchen.com/download/20180826SkinDL.pdf).

### data

The image data used to train, validate and test the network come from the open source ISIC Dermoscopic Archive. When selecting data, only images labeled as malignant or benign will be selected. The total data set consisted of 3183 images (1523 benign and 1660 malignant).

Data samples can be found in the DataSample folder, including 30 images (15 benign and 15 malignant).

### code

It is divided into three parts: Alexnet, Inception v3 and CNN-DIY, which correspond to three different convolutional neural networks in the experiment.

The code in the Alexnet and Inception v3 folders can be run by running a script with the corresponding folder name. Using Alexnet and Inception v3, training is done using transfer learning.

The code in the CNN-DIY folder can be run by running precision.m（InputSize为299\*299\*3） or precision\_2.m（InputSize为146\*146\*3）. We used a convolutional neural network designed ourself.


## 利用深度学习网络进行黑色素瘤检测的皮肤病变分析

这是“利用深度学习网络进行黑色素瘤检测的皮肤病变分析”的实验报告。

### data

用于训练，验证和测试网络的图像数据来自开源的ISIC Dermoscopic Archive。在选择数据时，仅选择标记为恶性（malignant）黑素瘤或良性（benign）痣的图像。总的数据集由3183个图像（1523个良性痣和1660个恶性黑素瘤）构成。

数据样本可以见DataSample文件夹，包括30个图像（15个良性痣和15个恶性黑素瘤）

### code

分为三部分：Alexnet，Inception v3和CNN-DIY，分别对应实验中三个不同的卷积神经网络。

Alexnet和Inception v3文件夹中的代码，可以运行对应文件夹名的脚本即可运行。使用的是Alexnet和Inception v3，利用转移学习的方式进行训练。

CNN-DIY文件夹中的代码，可以运行precision.m（InputSize为299\*299\*3）或者precision\_2.m（InputSize为146\*146\*3）即可运行。使用的是自己设计的卷积神经网络。



