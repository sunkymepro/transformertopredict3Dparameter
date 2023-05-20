import numpy as np
from os import listdir
import torch
import glob
from scipy.io import loadmat
import os


class data_deca(torch.utils.data.Dataset):
	def __init__(self):
		super(data_deca, self).__init__()
		self.n_videos = glob.glob(os.path.join('./deca/merge_smooth/Actor_1/*.mat'))#参数序列文件
		self.norm = loadmat('./deca/norm.mat')#整个数据集的六维参数的均值，方差，最大值，最小值，用于归一化然后线性变化到0-1
		self.data_list = {}
		n_videos = self.n_videos
		data_list = []
		for i in n_videos:#读取数据，对于每个mat文件，把文件读取成一个个40x6的数据，然后归一化并线性映射到0-1
			data = loadmat(i)
			pose = (data['pose'][:,:3] - self.norm['pose_mean'][:,:3])/np.sqrt(self.norm['pose_var'][:,:3])
			pose = (pose - self.norm['pose_min'][:,:3])/(self.norm['pose_max'][:,:3] - self.norm['pose_min'][:,:3])
			cam = (data['cam'] - self.norm['cams_mean'])/np.sqrt(self.norm['cams_var'])
			cam = (cam - self.norm['cams_min'])/(self.norm['cams_max'] - self.norm['cams_min'])
			data = np.concatenate([pose, cam], axis=1)
			for j in range(data.shape[0]-40):
				data_list.append(data[j:j+40,:][np.newaxis])
		self.data_list=np.concatenate(data_list, axis=0)

	def __getitem__(self, index):
		inputs = self.data_list[index]#从预先得到的数据列表中提取出对应index的数据（40x6），前20x6为src，后20x6为tgt
		input = torch.tensor(inputs[:20,:], dtype=torch.float32)
		input2 = torch.tensor(inputs[20:,:], dtype=torch.float32)
		out = {
			'input' : input.type(torch.float32),
			'input2' : input2.type(torch.float32),
		}
		return out
	def __len__(self):
		return self.data_list.shape[0]