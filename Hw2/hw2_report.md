# CNN Classification on CIFAR-100 Dataset
未央-机械01 金佳熠 2020012933

## Step 1: MLP Classification
在本节中，实现了完整的基于Pytorch的MLP代码用于图像分类。通过调整隐层的数量和宽度，计算模型的总参数量，观察模型分类准确率与上述参数的关系。

实验中超参数设置如下：

| Hyperparameters | Value | Meaning |
| --------------- | ----- | ------- |
| num_epochs | 20 | training epochs |
| lr | 0.05 | learning rate |
| batchsize | 256 | training batchsize |
| logging_steps | 100 | logging batchsize |

### 128宽的单隐层MLP
当使用参数为128宽的单隐层MLP时，模型总共有两层，总参数量为406244，经过20轮迭代后，测试集上最佳准确率为22.860%，训练集和验证集上的损失和准确率曲线如下图所示。

![](result/step1_1_loss.jpg)
损失曲线（黄线为测试集，蓝线为训练集，下同）

![](result/step1_1_acc.jpg)
准确率曲线（紫线为测试集，红线为训练集，下同）

### 256宽的单隐层MLP
调整隐层的宽度为256，模型总共有两层，总参数量为812388，经过20轮迭代后，测试集上最佳准确率为23.260%，训练集和验证集上的损失和准确率曲线如下图所示。

![](result/step1_2_loss.jpg)

![](result/step1_2_acc.jpg)