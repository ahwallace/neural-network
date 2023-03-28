import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

class nn:
	def __init__(self, X_train, y_train, hidden_layers, batch_size, learning_rate, max_iter, threshold, X_val = None, y_val = None, verbose = False) -> None:
		self.X_train = X_train
		self.y_train = y_train
		self.hidden_layers = hidden_layers
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.max_iter = max_iter
		self.threshold = threshold
		self.X_val = X_val
		self.y_val = y_val
		self.verbose = verbose

		self.n_inputs = X_train.shape[1] # Number of input features (nodes in input layer)
		self.n_outputs = 1 # Binary classification
		self.n_train_samples = len(X_train)
		self.layers = [self.n_inputs] + hidden_layers + [1]
		self.n_layers = len(self.layers) # input layer + hidden layer + output layer

		# Initialise weights and bias nodes for all layers from gaussian N(0, 0.01)
		self.weights = [np.random.normal(size=(self.layers[l+1], self.layers[l]), scale=1) for l in range(self.n_layers-1)] # Suggestion from Week 2 of notes - CT5133 (0.01 too small - weight changes tended to 0)
		self.biases = [np.zeros((self.layers[l+1], 1)) for l in range(self.n_layers-1)] # Suggestion from pg 173 of www.deeplearningbook.org

		if self.verbose:
			print(f'bias vectors: {[b.shape for b in self.biases]}')
			print(f'weight matrices: {[w.shape for w in self.weights]}')

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))

	def sigmoid_deriv(self, z):
		return self.sigmoid(z)*(1-self.sigmoid(z))

	def cost_func(self, y_hat, y):
		y = np.array(y)
		y_hat = np.array(y_hat)
		cost = np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
		return -cost

	def feedforward(self, x):
		'''
		x: vector of input values (a single training sample)
		'''

		zs = []
		activations = [x.reshape(-1, 1)]

		for l in range(self.n_layers-1):
			z = np.matmul(self.weights[l], activations[l]) + self.biases[l]
			zs.append(z)
			activations.append(self.sigmoid(z))
		return zs, activations
	
	def predict(self, X):
		y_hats = []
		for x in X:
			zs, activations = self.feedforward(x)
			y_hat = activations[-1][0][0]

			if y_hat > 0.5:
				y_hat = 1
			else:
				y_hat = 0
			
			y_hats.append(y_hat)
		
		return y_hats
	
	def fit(self):
		costs = [0]
		last_check = 0
		for i in range(self.max_iter):
			shuffled_index = np.random.permutation(self.n_train_samples)

			n_batches = np.floor(self.n_train_samples/self.batch_size)
			mini_batch_indices = np.array_split(shuffled_index, n_batches) # create mini batches of size batch_size

			epoch_cost = 0
			for index in mini_batch_indices:
				
				n_mini_batch = len(index)
				X_mini = self.X_train[index, ]
				y_mini = self.y_train[index, ]
				
				# Stores partial derivative calculated over current batch in data
				delta_w_deriv = [np.zeros(w.shape) for w in self.weights]
				delta_b_deriv = [np.zeros(b.shape) for b in self.biases]
				
				for j in range(n_mini_batch):
					x = X_mini[j]
					y = y_mini[j]

					zs, activations = self.feedforward(x) # Forward pass
					
					delta = activations[-1] - y # Final layer delta

					delta_w_deriv[-1] += np.matmul(delta, np.transpose(activations[-2]))
					delta_b_deriv[-1] += delta

					for l in range(2, self.n_layers): # start at layer L-1 and work backwards
						delta = np.matmul(np.transpose(self.weights[-l+1]), delta)*self.sigmoid_deriv(zs[-l])
						delta_w_deriv[-l] += np.matmul(delta, np.transpose(activations[-l-1]))
						delta_b_deriv[-l] += delta
					
				for l in range(self.n_layers-1):
					self.weights[l] -= (self.learning_rate/n_mini_batch)*delta_w_deriv[l]
					self.biases[l] -= (self.learning_rate/n_mini_batch)*delta_b_deriv[l]

				y_hats = []
				for x in X_mini:
					zs, activations = self.feedforward(x)
					y_hats.append(activations[-1][0][0])
					
				epoch_cost += np.abs(self.cost_func(y_hats, y_mini))

			costs.append(epoch_cost/n_batches)

			if i - last_check == 10:
				last_check = i
				if np.abs(costs[-1] - costs[-10]) <= self.threshold:
					print(f'Thresh: {np.abs(costs[-1] - costs[-2]):.3f}')
					break
		
		ys = self.predict(self.X_train)

		return ys, costs

if __name__ == 'main':
	np.random.seed(123)

	# MNIST Fashion dataset
	def load_mnist(path, kind='train'): 
		"""Load MNIST data from `path`"""

		labels_path = os.path.join(path,
								'%s-labels-idx1-ubyte.gz'
								% kind)
		images_path = os.path.join(path,
								'%s-images-idx3-ubyte.gz'
								% kind)

		with gzip.open(labels_path, 'rb') as lbpath:
			labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
								offset=8)

		with gzip.open(images_path, 'rb') as imgpath:
			images = np.frombuffer(imgpath.read(), dtype=np.uint8,
								offset=16).reshape(len(labels), 784)

		return images, labels

	train_imgs, train_labels = load_mnist('./fashion-mnist-data', 'train')
	test_imgs, test_labels = load_mnist('./fashion-mnist-data', 't10k')

	# 2 Classes for binary classification
	binary_labels = [2, 3]
	binary_label_names = ['Pullover', 'Dress']

	binary_train_index = [ind for ind, x in enumerate(train_labels) if x == binary_labels[0] or x == binary_labels[1]]
	binary_train_imgs = train_imgs[binary_train_index]
	binary_train_labels = train_labels[binary_train_index]

	binary_train_imgs, binary_val_imgs, binary_train_labels, binary_val_labels = train_test_split(binary_train_imgs, binary_train_labels, test_size=2000, shuffle = True, random_state = 100)

	binary_test_index = [ind for ind, x in enumerate(test_labels) if x == binary_labels[0] or x == binary_labels[1]]
	binary_test_imgs = test_imgs[binary_test_index]
	binary_test_labels = test_labels[binary_test_index]

	print(binary_train_imgs.shape)
	print(binary_val_imgs.shape)
	print(binary_test_imgs.shape)

	# Let class 2 == 0 and class 3 == 1
	binary_train_labels[np.where(binary_train_labels == binary_labels[0])] = 0
	binary_train_labels[np.where(binary_train_labels == binary_labels[1])] = 1

	binary_val_labels[np.where(binary_val_labels == binary_labels[0])] = 0
	binary_val_labels[np.where(binary_val_labels == binary_labels[1])] = 1

	binary_test_labels[np.where(binary_test_labels == binary_labels[0])] = 0
	binary_test_labels[np.where(binary_test_labels == binary_labels[1])] = 1

	nn_multi = nn(binary_train_imgs, binary_train_labels, [4, 4, 4], 12, 0.01, 1000, 0.0001, X_val = None, y_val = None, verbose = False)
	y_hat_train_multi_fashion, costs_train_multi_fashion = nn_multi.fit()

	y_hat_val_multi_fashion = nn_multi.predict(binary_val_imgs)

	print(f'Train set accuracy: {accuracy_score(binary_train_labels, y_hat_train_multi_fashion):.3f}')
	print(f'Validation set accuracy: {accuracy_score(binary_val_labels, y_hat_val_multi_fashion):.3f}')
	plt.plot(costs_train_multi_fashion[1:])

	# Fashion test set
	y_hat_test_multi_fashion = nn_multi.predict(binary_test_imgs)

	print(f'Test set accuracy: {accuracy_score(binary_test_labels, y_hat_test_multi_fashion):.3f}')
	print(confusion_matrix(binary_test_labels, y_hat_test_multi_fashion))