import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass

from torchvision import datasets # load MNIST
import torchvision.transforms as T # transformers for computer vision
from torch.optim.lr_scheduler import ReduceLROnPlateau 

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # progress bar


##### Preprocessing #####
totensor = T.ToTensor() # image (3D array) to Tensor

train_data = datasets.MNIST(root = './', download=False, train = True, transform = totensor)
test_data = datasets.MNIST(root = './', download=False, train = False, transform = totensor)

#  0, 1, 2 
new_train_img = []
new_train_label = []
for i in range(len(train_data)):
    img, label = train_data[i]
    if label == 0 or label == 1 or label == 2:
        new_train_img.append(img.numpy())
        new_train_label.append(label)

new_test_img = []
new_test_label = []
for i in range(len(test_data)):
    img, label = test_data[i]
    if label == 0 or label == 1 or label == 2:
        new_test_img.append(img.numpy())
        new_test_label.append(label)

new_train_img, new_train_label = np.array(new_train_img), np.array(new_train_label)
new_test_img, new_test_label = np.array(new_test_img), np.array(new_test_label)


##### Create dataset & dataloader #####
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass

class MNIST_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).type(torch.FloatTensor)
        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.n_samples = x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

torch.manual_seed(101)

train_loader = DataLoader(MNIST_Dataset(new_train_img, new_train_label), batch_size=100, shuffle=True)
test_loader =  DataLoader(MNIST_Dataset(new_test_img, new_test_label),  batch_size=500, shuffle=False)

@dataclass
class Loader:
    """
    Custom class to accomodate train and test loader iterators
    If we run one iteration now, we will have one batch of the training dataset
    """
    train: DataLoader
    test: DataLoader

mnist_loader = Loader(train_loader, test_loader)


##### Create ANN #####
class ThreeLayerANN(nn.Module):
    
    def __init__(self, in_size = 784, out_size=3):
        """
        * 784 input layers 
        * 2 hiden layers of 3 and 3 neurons respectively
        * 3 output layer
        """
        super(ThreeLayerANN, self).__init__()
        
        self.layer1 = nn.Linear(in_size, 3)
        self.layer2 = nn.Linear(3, 3)
        self.layer3 = nn.Linear(3, out_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)

        # weights initialize
        self.layer1.weight.data.normal_(mean=0.0, std=1.0)
        self.layer1.bias.data.zero_()
        self.layer2.weight.data.normal_(mean=0.0, std=1.0)
        self.layer2.bias.data.zero_()
        self.layer3.weight.data.normal_(mean=0.0, std=1.0)
        self.layer3.bias.data.zero_()
            
    def forward(self, X):

        X = self.sigmoid(self.layer1(X))
        X = self.sigmoid(self.layer2(X))
        X = self.layer3(X)
        
        return self.softmax(X) # multi-class classification, the sum of all probabilities is 1
    

##### Training function #####
def train_model(model, loader, criterion, optimizer, scheduler, epochs=50, synapse=False, weight_file='', w_range=(-1, 1)):
    """
    synapse : bool, default=False. synapse=True
    """
    if synapse:
        synapse_weight = np.load(weight_file)
    Loss = {'train':[], 'test':[]}
    Accuracy = {'train':[], 'test':[]}
    model = model.cuda()

    for epoch in tqdm(range(epochs)):
        # we set the number of True positives to zero in every epoch
        train_corr = 0
        test_corr = 0
        train_size = 0
        test_size = 0
        
        ##### Training loop #####
        model.train()
        for batch, (img, label) in enumerate(loader.train): 
            batch +=1
            train_size += len(img)
            img = img.cuda()
            label = label.cuda()
            y_pred = model(img.view(img.shape[0],-1)) # batch size for train is 100
            loss = criterion(y_pred, label)
            
            # last 3-layer neurons into one result
            _, prediction = torch.max(y_pred, dim=1) # return (values, indices)
            train_corr += (prediction == label).sum() # sum of correct predictions

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if synapse:
                #####  #####
                for p in model.parameters():
                    p.data.clamp_(w_range[0], w_range[1])
                #######################

                #####  #####
                with torch.no_grad():
                    for i, param in enumerate(model.parameters()):
                        param_shape = param.shape
                        param_clone = torch.clone(param).cpu()
                        w_index = np.argmin(np.abs(param_clone.numpy().reshape(-1, 1) - np.tile(np.array(synapse_weight).reshape(1, -1), (param_clone.numpy().size, 1))), axis=1)
                        param[...] = torch.Tensor(synapse_weight[w_index]).view(param_shape)
                #######################
        
        accuracy = 100 * (train_corr.item() / train_size ) 
        print( f'Epoch:{epoch+1:2d} Train Loss: {loss.item():4.4f} Train Accuracy: {accuracy:4.4f} %' )

        Loss['train'].append(loss.item()) # store loss at the end of epoch
        Accuracy['train'].append(accuracy) # store accuracy at the end of epoch

        ##### Validation loop #####
        model.eval()
        # validation (test). Here we run batches of 500 images, the test loader runs 120 times (120 x 500 = 60,000)
        with torch.no_grad():
            for batch, (img, label) in enumerate(loader.test):
                batch +=1
                test_size += len(img)
                img = img.cuda()
                label = label.cuda()
                y_val = model(img.view(img.shape[0],-1)) # batch size for test is 500
                _, predicted = torch.max(y_val, dim = 1)
                test_corr += (predicted == label).sum()
        
        loss = criterion(y_val, label)
        
        scheduler.step(loss)

        Loss['test'].append(loss.item()) 
        accuracy = 100 * (test_corr.item() / test_size)
        print( f'Test Accuracy: {accuracy:4.4f} %' )
        Accuracy['test'].append(accuracy)
    
    return model, Loss, Accuracy


# case2 training (ideal activation function / extracted synapse weight)
model_case2 = ThreeLayerANN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_case2.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, verbose=True, min_lr=1e-6)
model2, Loss2, Accuracy2 = train_model(model=model_case2, loader=mnist_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                                       epochs=100,
                                       synapse=True,
                                       weight_file='Synapse_weight\weight2_3period.npy',
                                       w_range=(-2, 2))

best_acc = np.max(Accuracy2['test'])
print('----------------------------------------------------')
print(f'Best Test Accuracy: {best_acc:4.4f} %')
