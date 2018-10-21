import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader

class LoadMnistTrain(Dataset):
	def __init__(self):
		df = pd.read_csv('./data/train.csv')
		self.len = df.shape[0]
		self.train = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32)
		self.label = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)

	def __getitem__(self, index):
		return self.train[index].view(1,28,28), self.label[index]

	def __len__(self):
		return self.len

class LoadMnistTest(Dataset):
	def __init__(self):
		df = pd.read_csv('./data/test.csv')
		self.len = df.shape[0]
		self.test = torch.tensor(df.iloc[:,:].values, dtype=torch.float32)

	def __getitem__(self, index):
		return self.test[index].view(1,28,28)

	def __len__(self):
		return self.len


if __name__ == '__main__':
	df = pd.read_csv('./data/train.csv')
	for i in range(1,4):
		plt.subplot(130+i)
		X = torch.Tensor(np.array(df.iloc[330+i][1:].values.reshape(1,28,28)))
		plt.imshow(X.data[0], cmap=plt.get_cmap('gray'))
		plt.title(df.iloc[330+i][0])
	plt.show()