# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch

#####
# Part 2 To-Do List:
# * Change activation function.
# * Use multiple activation functions (one between input to hidden, one between hidden and output)
# * Change number of hidden units
# * Add more hidden layers (with different number of hidden units). Makes more weights and biases.
# * Add more activation functions for these hidden layers
# * Implement L2 Regularization
# * Implement data standardization
#####

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size

        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn                          # loss function

        self.inputToHidden = torch.nn.Linear(in_size, 150)       # weight.size = [100, 3072]. bias.size = [100]
        self.tanh = torch.nn.ReLU()                       # Activation function
        self.hiddenToHidden = torch.nn.Linear(150, 100)
        self.hiddenActivation = torch.nn.LeakyReLU()
        self.hiddenToOutput = torch.nn.Linear(100, out_size)     # weight.size = [2, 32]. bias.size = [2]

        self.inputToHiddenOpt = torch.optim.Adam([self.inputToHidden.weight, self.inputToHidden.bias], lr=lrate)
        self.hiddenToOutputOpt = torch.optim.Adam([self.hiddenToOutput.weight, self.hiddenToOutput.bias], lr=lrate)
        self.hiddenToHiddenOpt = torch.optim.Adam([self.hiddenToHidden.weight, self.hiddenToHidden.bias], lr=lrate)

    def set_parameters(self, params):
        """ Set the parameters of your network
        @param params: a list of tensors containing all parameters of the network
        """
        self.inputToHidden.weight = params[1]
        self.inputToHidden.bias = params[2]
        self.hiddenToOutput.weight = params[3]
        self.hiddenToOutput.bias = params[4]
        self.hiddenToHidden.weight = params[5]
        self.hiddenToHidden.bias = params[6]
        return

    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        return [self.inputToHidden.weight, self.inputToHidden.bias, 
        self.hiddenToOutput.weight, self.hiddenToOutput.bias, 
        self.hiddenToHidden.weight, self.hiddenToHidden.bias]


    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        xNP = x.numpy()
        xMean = np.mean(xNP)
        xSTD = np.std(xNP)
        xNP = (xNP - xMean) / xSTD
        x = torch.from_numpy(xNP)
        y = torch.ones(x.shape[0], 2)
        for index, image in enumerate(x):
            tanInput = torch.matmul(self.inputToHidden.weight, image) + self.inputToHidden.bias
            tan = self.tanh(tanInput)
            yVal = torch.matmul(self.hiddenToHidden.weight, tan) + self.hiddenToHidden.bias
            activator = self.hiddenActivation(yVal)
            finalVal = torch.matmul(self.hiddenToOutput.weight, activator) + self.hiddenToOutput.bias
            y[index] = finalVal
        return y

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        self.inputToHiddenOpt.zero_grad()
        self.hiddenToHiddenOpt.zero_grad()
        self.hiddenToOutputOpt.zero_grad()
        yhats = self.forward(x)
        parameters = self.get_parameters()
        reg = torch.tensor(0.)
        for parameter in parameters:
            reg += torch.norm(parameter)
        loss = self.loss_fn(yhats, y) + (.0005 * reg)
        loss.backward()
        self.inputToHiddenOpt.step()
        self.hiddenToHiddenOpt.step()
        self.hiddenToOutputOpt.step()
        #print(loss)
        return loss.item()


# Taken from https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
class trainData(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

def fit(train_set,train_labels,dev_set,n_iter,batch_size=500):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, out_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of epochs of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of approximations to labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N
    """

    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []
    net = NeuralNet(.001, loss_fn, train_set.size()[1], 2)         # The input size is the length of features (3072). Output size is different possible classes (2).     
    training_data = trainData(train_set, train_labels)
    training_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
    for epoch in range(100 ):
        # training_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
        batch = next(iter(training_loader))
        batchStep = net.step(batch[0], batch[1])
        losses.append(batchStep)
    predictions = net.forward(dev_set)
    dev_labels = np.zeros(dev_set.size()[0])
    for index, pred in enumerate(predictions):
        dev_labels[index] = torch.argmax(pred).item()
    torch.save(net, "net")
    return losses, dev_labels, net
