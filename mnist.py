import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from dataloader import LoadMnistTrain, LoadMnistTest
from torch.utils.data import DataLoader

class mnist(nn.Module):
	def __init__(self):
		super(mnist, self).__init__()
		self.conv1 = nn.Conv2d(1, 4, 3)
		self.pool  = nn.MaxPool2d(2, stride=2)
		self.conv2 = nn.Conv2d(4, 12, 4)
		self.fc1 = nn.Linear(12 * 5 * 5, 240)
		self.fc2 = nn.Linear(240, 120)
		self.fc3 = nn.Linear(120, 84)
		self.fc4 = nn.Linear(84, 10)
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 12 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x


if __name__ == '__main__':
	trainset = LoadMnistTrain()
	train_loader = DataLoader(dataset = trainset,
	                      	  batch_size = 32,
	                      	  shuffle = True,
	                      	  num_workers = 2)

	testset = LoadMnistTest()
	test_loader = DataLoader(dataset = testset,
	                      	  batch_size = 32,
	                      	  shuffle = False,
	                      	  num_workers = 2)

	model = mnist()
	criterian = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

	for epoch in range(10):
		running_loss = 0.0
		for i, data in enumerate(train_loader, 0):
			image, label = data
			optimizer.zero_grad()
			out = model(image)
			loss = criterian(out, label)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			if i % 200 == 199:
				print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/200))
				running_loss = 0.0

	# show part of the test results
	dataiter = iter(test_loader)
	image = dataiter.next()
	output = model(image)
	_, predicted = torch.max(output.data, 1)
	for i, X in enumerate(image, 0):
		plt.subplot(4,8,i+1)
		plt.imshow(X.data[0], cmap=plt.get_cmap('gray'))
	plt.show()
	print(predicted)

	# predict
	results = torch.tensor((), dtype=torch.long)
	with torch.no_grad():
		for data in test_loader:
			outputs = model(data)
			_, predicted = torch.max(outputs.data, 1)
			results = torch.cat((results, predicted), 0)
	
	# Generate submission file
	results = pd.Series(results, name='Label')
	submission = pd.concat([pd.Series(range(1, 28001), name='ImageId'), results], axis=1)
	submission.to_csv('pytorch_mnist.csv', index=False)

