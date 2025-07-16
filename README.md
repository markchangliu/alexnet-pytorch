# AlexNet-Pytorch

复现2012年论文《ImageNet Classification with Deep Convolutional Neural Networks》.

## 数据

Tiny-ImageNet, 来源`https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet`.

## 训练、测试、推理

训练: 代码参考`demos/demo_train.py`, 训练超参数`demos/train_cfg.json`.

测试: 代码参考`demos/demo_test.py`, 测试配置参考`demos/test_cfg.json`.

推理: 代码参考`demos/demo_infer.ipynb`

## 最佳ckpt测试结果

|     | Test Accuracy | Test Loss |
| -------- | ------- | ------- |
| no TTA  | 0.377    | 2.945 |
| TTA | 0.411     | 2.582 |
