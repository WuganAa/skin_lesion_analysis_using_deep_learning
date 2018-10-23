这是“利用深度学习网络进行黑色素瘤检测的皮肤病变分析”实验报告的代码。
分为三部分：Alexnet，Inception v3和CNN-DIY，分别对应实验中三个不同的卷积神经网络。

用于训练，验证和测试网络的图像数据来自开源的ISIC Dermoscopic Archive。
在选择数据时，仅选择标记为恶性（malignant）黑素瘤或良性（benign）痣的图像。
总的数据集由3183个图像（1523个良性痣和1660个恶性黑素瘤）构成。
数据样本可以见DataSample文件夹，包括30个图像（15个良性痣和15个恶性黑素瘤）

Alexnet和Inception v3文件夹中的代码，可以运行对应文件夹名的脚本即可运行。
使用的是Alexnet和Inception v3，利用转移学习的方式进行训练。

CNN-DIY文件夹中的代码，可以运行precision.m（InputSize为299*299*3）
或者precision_2.m（InputSize为146*146*3）即可运行。
使用的是自己设计的卷积神经网络。