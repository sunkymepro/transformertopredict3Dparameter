# transformertopredict3Dparameter
借鉴bert的思想来重建预测3D参数的代码

使用数据集为RAVDESS数据集，经过数据预处理之后，再用deca提取每一帧人脸的pose参数（3维）和cam参数（3维）

dataset.py为定义数据集的文件
train.py为主要文件
encoder.py定义了所用到的模型
soft_dtw_cuda.py为soft dynamic time warping loss
