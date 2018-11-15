#### 1 val 0.873   testA 0.87683

使denseblock3起的部分可以训练

dropout 0.2

数据增强

```python
trans_train = transforms.Compose([transforms.RandomResizedCrop(size=224),
                            transforms.RandomHorizontalFlip(),
#                           transforms.ColorJitter(0.5,0, 0.5,0),
                            transforms.RandomGrayscale(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

trans_valid = transforms.Compose([transforms.Resize(size=256),
                            transforms.CenterCrop(size=224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
```

<br>

#### 2 val 0.8731 dropout 0.2

随机种子34 

使denseblock3起的部分可以训练

```python
trans_train = transforms.Compose([transforms.RandomResizedCrop(size=224),
                            transforms.RandomHorizontalFlip(),
#                             transforms.ColorJitter(0.5,0, 0.5,0),
                            transforms.RandomGrayscale(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.47954108864506007, 0.5295650244021952, 0.39169756009537665],
                                 std=[0.21481591229053462, 0.20095268035289796, 0.24845895286079178])])

trans_valid = transforms.Compose([transforms.Resize(size=256),
                            transforms.CenterCrop(size=224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.47954108864506007, 0.5295650244021952, 0.39169756009537665],
                                 std=[0.21481591229053462, 0.20095268035289796, 0.24845895286079178])])
```

#### 3 使用30%的子数据集进行label smoothing的测试，三种损失函数的结果

`nn.MultiLabelSoftMarginLoss()`  0.857048

`nn.BCEWithLogitsLoss()`   0.852643

`def cross_entropy(input, target, size_average=True):` 0.858370



#### 4 Densenet201双GPU，label_smoothing = 0.1，0.8757

[Epoch 9] train loss 1.021871 train acc 0.870137  valid loss 1.023163 valid acc 0.868943  time 402.153438

[Epoch 10] train loss 1.017513 train acc 0.874267  valid loss 1.020073 valid acc 0.874670  time 400.856712

[Epoch 11] train loss 1.012702 train acc 0.874645  valid loss 1.017438 valid acc 0.874009  time 400.136600

[Epoch 12] train loss 1.010013 train acc 0.876379  valid loss 1.017918 valid acc 0.874009  time 400.168997

[Epoch 13] train loss 1.006022 train acc 0.877830  valid loss 1.017303 valid acc 0.873348  time 397.005429

[Epoch 14] train loss 1.003956 train acc 0.878239  valid loss 1.019049 valid acc 0.873128  time 403.903243
Finished Training
best_epoch: 8, best_val_acc 0.875771

数据增强：

```
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

dataset_train = MyDataset(df_data=train, 
    data_dir=train_path, transform=trans_train)
dataset_valid = MyDataset(df_data=val, 
    data_dir=val_path, transform=trans_valid)

loader_train = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
loader_valid = DataLoader(dataset = dataset_valid, batch_size=32, shuffle=False, num_workers=0)

```

损失函数cross_entropy：

```
def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
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



#### 5.Densenet161 ，dropout 0.5 ，无label smooth，0.8733

![1541147190023](C:\Users\dxz\AppData\Roaming\Typora\typora-user-images\1541147190023.png)

  增强方式，优化器同上



#### 6.Densenet201,增强方式imgaug, 其他同上，0.870



#### 7.SE-Densenet161, label_smoothing,增强同4，BS = 64，dropout = 0.5，0.874670

[Epoch 20] train loss 1.022191 train acc 0.873100  valid loss 1.024608 valid acc 0.870264  time 469.689808
................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
[Epoch 21] train loss 1.019254 train acc 0.874046  valid loss 1.020659 valid acc 0.874670  time 468.956556
save model...
saved.
................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
[Epoch 22] train loss 1.024337 train acc 0.870736  valid loss 1.019694 valid acc 0.872467  time 472.815843
................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
[Epoch 23] train loss 1.024507 train acc 0.873447  valid loss 1.020893 valid acc 0.870705  time 473.257113
................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
[Epoch 24] train loss 1.026460 train acc 0.871902  valid loss 1.024001 valid acc 0.871806  time 472.365721
Finished Training
best_epoch: 21, best_val_acc 0.874670
