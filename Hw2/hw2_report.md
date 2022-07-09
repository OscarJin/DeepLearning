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
当使用参数为128宽的单隐层MLP时，