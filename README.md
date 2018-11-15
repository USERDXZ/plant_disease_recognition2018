# 2018-农作物病害检测比赛

B榜：0.88190          名次：54/138

[赛题链接](https://challenger.ai/competition/pdr2018)

### 数据分析

训练集31718张图片，包含10种植物的61种病害情况，测试集4320张图片

### 环境

- ubuntu16.04/windows10
- python 3.6.2
- PyTorch 0.4
- opencv-python 3.4
- imgaug 0.2.5

### 使用

- Densenet201.ipynb——用PyTorch Densenet201模型，进行fine-tune训练
- SE-Densenet161.ipynb——用PyTorch Densenet161作为基础模型，加入SE模块，进行fine-tune训练
- test bch.ipynb——对测试集预测
- test of random vote.ipynb——多次测试集预测，再进行投票
- sedensenet.py——SE-Densenet模型API
- imgaug.py——使用imgaug进行数据增强

### Model

1.PyTorch Densenet201 fine-tune，最后的线性层和最后两个denseblock可训练

2.PyTorch Densenet161 fine-tune，并加入SE模块

### Data augmentation

```python
batch_size = 128

trans_train = transforms.Compose([transforms.RandomResizedCrop(size=224),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomRotation(30),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.47954108864506007, 0.5295650244021952, 0.39169756009537665],
                                                       std=[0.21481591229053462, 0.20095268035289796, 0.24845895286079178])])

trans_valid = transforms.Compose([transforms.Resize(size=224),
                                  transforms.CenterCrop(size=224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.47954108864506007, 0.5295650244021952, 0.39169756009537665],
                                                       std=[0.21481591229053462, 0.20095268035289796, 0.24845895286079178])])
```

### Training

```python
def cross_entropy(input, target, size_average=True):
    logsoftmax = nn.LogSoftmax()
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

criterion = cross_entropy
optimizer = optim.Adam(params_to_update)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
label_smoothing = 0.1
```

详细训练过程参考 `train.md`

### 总结

1.自己计算mean，std，RandomResizedCrop，RandomRotation，稍有提升

2.加入label_smoothing = 0.1，换用cross_entropy，稍有提升

3.imgaug增强，提升不明显，训练速度变缓比较明显（打开方式不对）

4.SE模块提升不明显，训练速度变缓10%（打开方式不对）