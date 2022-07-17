# Deep CNN for Classifications on CIFAR-100 Dataset
未央-机械01 金佳熠 2020012933

## Step 1: Depth of CNN
在本节中通过调整CNN 的深度，观察模型的层数与最终的测试准确率的关系。保持模型超参数设置不变，如下表所示。

| Parameter | Value | Meaning |
| --------- | ------| -------|
| num_epochs | 50 | training epochs |
| lr | 0.1 | learning rate|
| batchsize | 256 | training batchsize |
| mmt | 0.9 | momentum for optimizer |
| wd | 5e-4 | weight decay for optimizer |
| data_augmentation | True | data augmentation |
| use_BN | True | enable batch normalization |

网络结构上，kernel size每层均为3，pooling stride每层均为2，pooling kernel size在最后十层为1、2相间，其余为1（对于1层CNN，pooling kernel size为2；对于5层CNN，pooling kernel size为1，1，2，1，2）。保持每层的channel数为32不变，观察CNN 层数从1、5、10、20、50、75、100 变化时，模型在训练收敛后在训练集和验证集上的准确率的变化。

### 100层CNN
模型最终测试准确率为。训练集和验证集上的损失和准确率曲线如下图所示。


损失曲线（黄线为验证集，蓝线为训练集，下同）


准确率曲线（紫线为验证集，红线为训练集，下同）