#%load_ext autoreload
#%autoreload 2
import copy
import math
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch import linalg as LA
from torchsummary import summary
from torch.nn.utils import weight_norm
def image_load_norm(imagepath):
    image = Image.open(imagepath)
    image = image.convert('L')
    mean = np.mean(image)
    scaled_image = (image - mean) / np.max(image)
    transform = transforms.ToTensor()
    image_tensor = transform(scaled_image)
    #image_norm_=(image_tensor / LA.vector_norm(image_tensor, ord=2))
    flattened_image=torch.flatten(image_tensor)
    return flattened_image


def sample_input_( radius,dim):
    x = torch.rand(1,dim)
    return (x / LA.vector_norm(x, ord=2))*radius
def generate_x_training_points(radius,dim,n):
    points=(sample_input_(radius,dim))
    for i in range (0,n-1):
        points=torch.cat((points,sample_input_(radius,dim)))
    return points
#2 layer
class Autoencoder_2_layers(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder_2_layers, self).__init__()
#super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.hidden_dim = hidden_dim

        self._initialize_weights()

    def forward(self, x_):
        x_ = self.sigmoid(1/np.sqrt(self.hidden_dim ) * self.fc1(x_))
        x_ = 1/np.sqrt(self.hidden_dim ) *self.fc2(x_)#32 for ex3 rest 1000
      #  x_ = self.sigmoid(x_)
        return x_

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
             #   m.weight.data *= 1 / np.sqrt(m.weight.size(1))
#3 layer
class Autoencoder_3_layers(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder_3_layers, self).__init__()
#super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self._initialize_weights()

    def forward(self, x_):
        x_ = self.sigmoid(1/np.sqrt(self.hidden_dim ) * self.fc1(x_))
        x_ = 1/np.sqrt(self.hidden_dim ) *self.fc2(x_)
        x_ = self.sigmoid(x_)
        x_ = 1/np.sqrt(self.input_dim ) *self.fc3(x_)
        return x_

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
                m.weight.data *= 1 / np.sqrt(m.weight.size(1))

#3 layer
class Autoencoder_4_layers(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder_4_layers, self).__init__()
#super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc4 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.hidden_dim = hidden_dim

        self._initialize_weights()

    def forward(self, x_):
        x_ = self.sigmoid(1/np.sqrt(self.hidden_dim) * self.fc1(x_))
        x_ = 1/np.sqrt(self.hidden_dim) *self.fc2(x_)
        x_ = self.sigmoid(x_)
        x_ = 1/np.sqrt(self.hidden_dim) *self.fc3(x_)
        x_ = self.sigmoid(x_)
        x_ = 1/np.sqrt(32) *self.fc4(x_)

        return x_

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
                m.weight.data *= 1 / np.sqrt(m.weight.size(1))
class Autoencoder_2_layers_(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder_2_layers_, self).__init__()
#super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self._initialize_weights()

    def forward(self, x_):
        x_ = self.sigmoid(1/np.sqrt(self.input_dim) * self.fc1(x_))
        x_ = 1/np.sqrt(self.hidden_dim) *self.fc2(x_)
      #  x_ = self.sigmoid(x_)
        return x_

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
               # m.weight.data *= 1 / np.sqrt(m.weight.size(1))

#dropout networks
class Autoencoder_4_layers_dropout(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder_4_layers_dropout, self).__init__()
#super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc4 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.1)
        self._initialize_weights()

    def forward(self, x_):
        x_ = self.sigmoid(1/np.sqrt(self.hidden_dim) * self.fc1(x_))
        x_ = self.dropout(x_)
        x_ = 1/np.sqrt(self.hidden_dim) *self.fc2(x_)
        x_ = self.sigmoid(x_)
        x_ = self.dropout(x_)
        x_ = 1/np.sqrt(self.hidden_dim) *self.fc3(x_)
        x_ = self.sigmoid(x_)
        x_ = self.dropout(x_)
        x_ = 1/np.sqrt(32) *self.fc4(x_)

        return x_

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
                m.weight.data *= 1 / np.sqrt(m.weight.size(1))
class Autoencoder_2_layers_dropout(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder_2_layers_dropout, self).__init__()
#super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout = nn.Dropout(0.1)

        self._initialize_weights()

    def forward(self, x_):
        x_ = self.sigmoid(1/np.sqrt(self.input_dim) * self.fc1(x_))
        x_ = self.dropout(x_)
        x_ = 1/np.sqrt(self.hidden_dim) *self.fc2(x_)
      #  x_ = self.sigmoid(x_)
        return x_

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
               # m.weight.data *= 1 / np.sqrt(m.weight.size(1))
class Autoencoder_3_layers_dropout(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder_3_layers_dropout, self).__init__()
#super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.1)
        self._initialize_weights()

    def forward(self, x_):
        x_ = self.sigmoid(1/np.sqrt(self.hidden_dim ) * self.fc1(x_))
        x_ = 1/np.sqrt(self.hidden_dim ) *self.fc2(x_)
        x_ = self.sigmoid(x_)
        x_ = self.dropout(x_)
        x_ = 1/np.sqrt(32) *self.fc3(x_)
        return x_

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
                m.weight.data *= 1 / np.sqrt(m.weight.size(1))

#3 layer
class Autoencoder_4_layers_tanh(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder_4_layers_tanh, self).__init__()
#super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc4 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.tanh = nn.Tanh()

        self._initialize_weights()

    def forward(self, x_):
        x_ = self.tanh(1/np.sqrt(1000) * self.fc1(x_))
        x_ = 1/np.sqrt(1000) *self.fc2(x_)
        x_ = self.tanh(x_)
        x_ = 1/np.sqrt(1000) *self.fc3(x_)
        x_ = self.tanh(x_)
        x_ = 1/np.sqrt(32) *self.fc4(x_)

        return x_

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
                m.weight.data *= 1 / np.sqrt(m.weight.size(1))
class Autoencoder_2_layers_tanh(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder_2_layers_tanh, self).__init__()
#super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.tanh = nn.Tanh()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self._initialize_weights()

    def forward(self, x_):
        x_ = self.tanh(1/np.sqrt( self.input_dim) * self.fc1(x_))
        x_ = 1/np.sqrt(self.hidden_dim) *self.fc2(x_)#32 for ex3 rest 1000
      #  x_ = self.sigmoid(x_)
        return x_

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
               # m.weight.data *= 1 / np.sqrt(m.weight.size(1))
class Autoencoder_3_layers_tanh(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder_3_layers_tanh, self).__init__()
#super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.tanh = nn.Tanh()

        self._initialize_weights()

    def forward(self, x_):
        x_ = self.tanh(1/np.sqrt(1000) * self.fc1(x_))
        x_ = 1/np.sqrt(1000) *self.fc2(x_)
        x_ = self.tanh(x_)
        x_ = 1/np.sqrt(32) *self.fc3(x_)
        return x_

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
                m.weight.data *= 1 / np.sqrt(m.weight.size(1))
class Autoencoder_2_layers_erf(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder_2_layers_erf, self).__init__()
#super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.erf = torch.erf
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self._initialize_weights()

    def forward(self, x_):
        x_ = self.erf(1/np.sqrt( self.input_dim) * self.fc1(x_))
        x_ = 1/np.sqrt(self.hidden_dim) *self.fc2(x_)#32 for ex3 rest 1000
      #  x_ = self.sigmoid(x_)
        return x_

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
               # m.weight.data *= 1 / np.sqrt(m.weight.size(1))

def make_diagonal(weights_1,weights_2,weights_3,input_,layers):
    if layers==2:
        temp = torch.mul(weights_1,1/np.sqrt(1000))
        w_alpha = torch.matmul(temp,input_.T)
        sigmoid = torch.sigmoid(w_alpha)
        sigmoid_derivative = sigmoid*(1-sigmoid)
        diag = torch.diagflat(sigmoid_derivative)
        return diag
    if layers==3:
        hiden_dim=1000
        temp = torch.mul(weights_1,1/np.sqrt(hiden_dim))
        w_alpha = torch.matmul(temp,input_.T)
        sigmoid = torch.sigmoid(w_alpha)
        sigmoid_ =torch.matmul( torch.mul(1/np.sqrt(hiden_dim),weights_2),sigmoid)
        sigmoid_derivative = sigmoid_*(1-sigmoid_)
        diag = torch.diagflat(sigmoid_derivative)
        return diag
    if layers==4:
        temp = torch.mul(weights_1,1/np.sqrt(1000))
        w_alpha = torch.matmul(temp,input_.T)
        sigmoid = torch.sigmoid(w_alpha)
        sigmoid_ = torch.matmul( torch.mul(1/np.sqrt(1000),weights_2),sigmoid)
        sigmoid_2 = torch.sigmoid(sigmoid_)
        sigmoid_2_ = torch.matmul(torch.mul(1/np.sqrt(1000),weights_3),sigmoid_2)
        sigmoid_derivative = sigmoid_2_*(1-sigmoid_2_)
        diag = torch.diagflat(sigmoid_derivative)
        return diag
def calculate_jacobian_2_layers(init_weights,trained_weights,input):
    differences_2_layers=[]
    weights_init_layer_0 = init_weights["fc1.weight"]
    weights_init_layer_1 = init_weights["fc2.weight"]
    diag=make_diagonal(weights_init_layer_0,None,None,input,2)
    init_Jacobian_2_layers =1/np.sqrt(1000)*weights_init_layer_1@diag@(1/np.sqrt(1000)*weights_init_layer_0)
    #init_Jacobian_2_layers=1/4 *(1/np.sqrt(32)* weights_init_layer_1)@(weights_init_layer_0*1/np.sqrt(1000))

    _, singular_values, _ = torch.svd(init_Jacobian_2_layers)
    operator_norm_init_2_layers = torch.max(singular_values)
    #print("Operator Norm of Initial Jacobian of 2 layers:", operator_norm_init_2_layers.item())
    eigenvalues_init_2_layers_ = torch.norm(torch.max(torch.abs(torch.linalg.eig(init_Jacobian_2_layers)[0]))*init_Jacobian_2_layers)
    eigenvalues_init_2_layers = torch.max(torch.abs(torch.linalg.eig(init_Jacobian_2_layers)[0]))
    #print("largest eigenvalue_: ", eigenvalues_init_2_layers_)

   # print("largest eigenvalue: ", eigenvalues_init_2_layers)
    ###trained Jacobian
    weights_train_layer_0 = trained_weights["fc1.weight"]
    weights_train_layer_1 = trained_weights["fc2.weight"]
    diag_train = make_diagonal(weights_train_layer_0,None,None,input,2)
    train_Jacobian_2_layers = 1/np.sqrt(1000)*weights_train_layer_1@diag_train@(1/np.sqrt(1000)*weights_train_layer_0)
    #train_Jacobian_2_layers=1/4 *1/np.sqrt(32)* weights_train_layer_1@weights_train_layer_0

    _, singular_values_, _ = torch.svd(train_Jacobian_2_layers)
    operator_norm_train_2_layers = torch.max(singular_values_)
    #print("Operator Norm of Trained Jacobian of 2 layers:", operator_norm_train_2_layers.item())
    eigenvalues_train_2_layers_ = torch.norm(torch.max(torch.abs(torch.linalg.eig(train_Jacobian_2_layers)[0]))*train_Jacobian_2_layers)
    eig_vals=torch.linalg.eig(train_Jacobian_2_layers)[0]
    eigenvalues_train_2_layers = torch.max(torch.abs(torch.linalg.eig(train_Jacobian_2_layers)[0]))

    #print("largest eigenvalue: ", eigenvalues_train_2_layers)
   # print("largest eigenvalue_: ", eigenvalues_train_2_layers_)

    diff=torch.abs(eigenvalues_train_2_layers-eigenvalues_init_2_layers)
   # diff_=torch.abs(eigenvalues_train_2_layers_-eigenvalues_init_2_layers_)

   # print("difference Between largest eigenvalue norms at init. and after training : ",diff )
   # print("difference Between largest eigenvalue norms at init. and after training : ",diff )

    differences_2_layers.append(diff)
    return eig_vals,differences_2_layers
def calculate_jacobian_3_layers(init_weights,trained_weights,input):
    differences_3_layers=[]

    weights_init_layer_0 = init_weights["fc1.weight"]
    weights_init_layer_1 = init_weights["fc2.weight"]
    hiden_dim = 1000
    input_dim=32
   # hidden_dim=1000
    #input_dim=32
    weights_init_layer_2 = init_weights["fc3.weight"]

    diag_2_init=make_diagonal(weights_init_layer_0,weights_init_layer_1,None,input,3)
    diag_1_init=make_diagonal(weights_init_layer_0,None,None,input,2)
    init_Jacobian_3_layers = 1/np.sqrt(hiden_dim)*weights_init_layer_2@diag_2_init@weights_init_layer_1@diag_1_init@weights_init_layer_0
    #print(init_Jacobian_3_layers)
    _, singular_values, _ = torch.svd(init_Jacobian_3_layers)
    operator_norm_init_2_layers = torch.max(singular_values)
   # print("Operator Norm of Initial Jacobian of 3 layers:", operator_norm_init_2_layers.item())
    eigenvalues_init_3_layers = torch.norm((torch.max(torch.abs(torch.linalg.eig(init_Jacobian_3_layers)[1])))*init_Jacobian_3_layers)
   # print("largest eigenvalue: ", eigenvalues_init_3_layers)
    ###trained Jacobian
    weights_train_layer_0 = trained_weights["fc1.weight"]
    weights_train_layer_1 = trained_weights["fc2.weight"]
    weights_train_layer_2 = trained_weights["fc3.weight"]
    diag_2_train = make_diagonal(weights_train_layer_0,weights_train_layer_1,None,input,3)
    diag_1_train = make_diagonal(weights_train_layer_0,None,None,input,2)
    train_Jacobian_3_layers = 1/np.sqrt(hiden_dim)*weights_train_layer_2@diag_2_train@(1/np.sqrt(hiden_dim)*weights_train_layer_1)@diag_1_train@(1/np.sqrt(input_dim)*weights_train_layer_0)
    _, singular_values_, _ = torch.svd(train_Jacobian_3_layers)
    operator_norm_train_3_layers = torch.max(singular_values_)
   # print("Operator Norm of Trained Jacobian of 3 layers:", operator_norm_train_3_layers.item())
    eigenvalues_train_3_layers =torch.norm( (torch.max(torch.abs(torch.linalg.eig(train_Jacobian_3_layers)[0])))*train_Jacobian_3_layers)
   # print("largest eigenvalue: ", eigenvalues_train_3_layers)
    diff=torch.abs(eigenvalues_train_3_layers-eigenvalues_init_3_layers)
  #  print("difference Between largest eigenvalue norms at init. and after training : ",diff )
    differences_3_layers.append(diff)
    return differences_3_layers
def calculate_jacobian_4_layers(init_weights,trained_weights,input):
    differences_4_layers=[]
    weights_init_layer_0 = init_weights["fc1.weight"]
    weights_init_layer_1 = init_weights["fc2.weight"]
    weights_init_layer_2 = init_weights["fc3.weight"]
    weights_init_layer_3 = init_weights["fc4.weight"]

    diag_1_init=make_diagonal(weights_init_layer_0,None,None,input,2)
    diag_2_init=make_diagonal(weights_init_layer_0,weights_init_layer_1,None,input,3)
    diag_3_init=make_diagonal(weights_init_layer_0,weights_init_layer_1,weights_init_layer_2,input,4)
    init_Jacobian_4_layers = 1/np.sqrt(1000)*weights_init_layer_3@diag_3_init@weights_init_layer_2@diag_2_init@weights_init_layer_1@diag_1_init@weights_init_layer_0
    _, singular_values, _ = torch.svd(init_Jacobian_4_layers)
    operator_norm_init_2_layers = torch.max(singular_values)
   # print("Operator Norm of Initial Jacobian of 3 layers:", operator_norm_init_2_layers.item())
    eigenvalues_init_4_layers = torch.norm((torch.max(torch.abs(torch.linalg.eig(init_Jacobian_4_layers)[0])))*init_Jacobian_4_layers)
  #  print("largest eigenvalue: ", eigenvalues_init_4_layers)
    weights_train_layer_0 = trained_weights["fc1.weight"]
    weights_train_layer_1 = trained_weights["fc2.weight"]
    weights_train_layer_2 = trained_weights["fc3.weight"]
    weights_train_layer_3 = trained_weights["fc4.weight"]
    diag_1_train = make_diagonal(weights_train_layer_0,None,None,input,2)
    diag_2_train = make_diagonal(weights_train_layer_0,weights_train_layer_1,None,input,3)
    diag_3_train = make_diagonal(weights_train_layer_0,weights_train_layer_1,weights_train_layer_2,input,4)
    train_Jacobian_4_layers = 1/np.sqrt(1000)*weights_train_layer_3@diag_3_train@weights_train_layer_2@diag_2_train@(1/np.sqrt(1000)*weights_train_layer_1)@diag_1_train@weights_train_layer_0
    _, singular_values_, _ = torch.svd(train_Jacobian_4_layers)
    operator_norm_train_4_layers = torch.max(singular_values_)
   # print("Operator Norm of Trained Jacobian of 4 layers:", operator_norm_train_4_layers.item())
    eigenvalues_train_4_layers =torch.norm( (torch.max(torch.abs(torch.linalg.eig(train_Jacobian_4_layers)[1])))*train_Jacobian_4_layers)
  #  print("largest eigenvalue: ", eigenvalues_train_4_layers)
    diff=torch.abs(eigenvalues_train_4_layers-eigenvalues_init_4_layers)
   # print("difference Between largest eigenvalue norms at init. and after training : ",diff )
    differences_4_layers.append(diff)
    return differences_4_layers
def make_diagonal_(weights_1,weights_2,weights_3,input_,layers):
    if layers==2:
        hidden_dim=weights_1.shape[1]

        temp = torch.mul(weights_1,1/np.sqrt(hidden_dim))
        w_alpha = torch.matmul(temp,input_.T)
        sigmoid = torch.sigmoid(w_alpha)
        sigmoid_derivative = sigmoid*(1-sigmoid)
        diag = torch.diagflat(sigmoid_derivative)
        return diag
    if layers==3:
        temp = torch.mul(weights_1,1/np.sqrt(1000))
        w_alpha = torch.matmul(temp,input_.T)
        sigmoid = torch.sigmoid(w_alpha)
        sigmoid_ =torch.matmul( torch.mul(1/np.sqrt(1000),weights_2),sigmoid)
        sigmoid_derivative = sigmoid_*(1-sigmoid_)
        diag = torch.diagflat(sigmoid_derivative)
        return diag
    if layers==4:
        temp = torch.mul(weights_1,1/np.sqrt(1000))
        w_alpha = torch.matmul(temp,input_.T)
        sigmoid = torch.sigmoid(w_alpha)
        sigmoid_ = torch.matmul( torch.mul(1/np.sqrt(1000),weights_2),sigmoid)
        sigmoid_2 = torch.sigmoid(sigmoid_)
        sigmoid_2_ = torch.matmul(torch.mul(1/np.sqrt(1000),weights_3),sigmoid_2)
        sigmoid_derivative = sigmoid_2_*(1-sigmoid_2_)
        diag = torch.diagflat(sigmoid_derivative)
        return diag
def make_diagonal_tanh(weights_1, weights_2, weights_3, input_, layers):
        if layers == 2:
            hidden_dim = weights_1.shape[1]
            temp = torch.mul(weights_1, 1 / np.sqrt(hidden_dim))
            w_alpha = torch.matmul(temp, input_.T)
            #sigmoid = torch.sigmoid(w_alpha)
            tanh_derivative = 1 - torch.tanh(w_alpha)**2
            diag = torch.diagflat(tanh_derivative)
            return diag
        if layers == 3:
            temp = torch.mul(weights_1, 1 / np.sqrt(1000))
            w_alpha = torch.matmul(temp, input_.T)
            tanh = torch.tanh(w_alpha)
            tanh_ = torch.matmul(torch.mul(1 / np.sqrt(1000), weights_2), tanh)
            tanh_derivative = 1 - torch.tanh(tanh_)**2
            diag = torch.diagflat(tanh_derivative)
            return diag
        if layers == 4:
            temp = torch.mul(weights_1, 1 / np.sqrt(1000))
            w_alpha = torch.matmul(temp, input_.T)
            tanh = torch.tanh(w_alpha)
            tanh_ = torch.matmul(torch.mul(1 / np.sqrt(1000), weights_2), tanh)
            tanh_1 = torch.tanh(tanh_)
            tanh_2 = torch.matmul(torch.mul(1 / np.sqrt(1000), weights_3), tanh_1)
            tanh_derivative = 1 - torch.tanh(tanh_2)**2
            diag = torch.diagflat(tanh_derivative)
            return diag
def calculate_jacobian_2_layers_tanh(init_weights,trained_weights,input):
    differences_2_layers=[]

    weights_init_layer_0 = init_weights["fc1.weight"]
    weights_init_layer_1 = init_weights["fc2.weight"]
    hidden_dim=init_weights["fc1.weight"].shape[0]
    input_dim=init_weights["fc1.weight"].shape[1]
    diag=make_diagonal_tanh(weights_init_layer_0,None,None,input,2)
    init_Jacobian_2_layers =1/np.sqrt(hidden_dim)*weights_init_layer_1@diag@(1/np.sqrt(hidden_dim)*weights_init_layer_0)
    #init_Jacobian_2_layers=1/4 *(1/np.sqrt(32)* weights_init_layer_1)@(weights_init_layer_0*1/np.sqrt(1000))

    _, singular_values, _ = torch.svd(init_Jacobian_2_layers)
    operator_norm_init_2_layers = torch.max(singular_values)
    #print("Operator Norm of Initial Jacobian of 2 layers:", operator_norm_init_2_layers.item())
    eigenvalues_init_2_layers_ = torch.norm(torch.max(torch.abs(torch.linalg.eig(init_Jacobian_2_layers)[0]))*init_Jacobian_2_layers)
    eigenvalues_init_2_layers = torch.max(torch.abs(torch.linalg.eig(init_Jacobian_2_layers)[0]))
    #print("largest eigenvalue_: ", eigenvalues_init_2_layers_)

   # print("largest eigenvalue: ", eigenvalues_init_2_layers)
    ###trained Jacobian
    weights_train_layer_0 = trained_weights["fc1.weight"]
    weights_train_layer_1 = trained_weights["fc2.weight"]
    diag_train = make_diagonal_tanh(weights_train_layer_0,None,None,input,2)
    train_Jacobian_2_layers = 1/np.sqrt(hidden_dim)*weights_train_layer_1@diag_train@(1/np.sqrt(hidden_dim)*weights_train_layer_0)
    #train_Jacobian_2_layers=1/4 *1/np.sqrt(32)* weights_train_layer_1@weights_train_layer_0

    _, singular_values_, _ = torch.svd(train_Jacobian_2_layers)
    operator_norm_train_2_layers = torch.max(singular_values_)
    #print("Operator Norm of Trained Jacobian of 2 layers:", operator_norm_train_2_layers.item())
    eigenvalues_train_2_layers_ = torch.norm(torch.max(torch.abs(torch.linalg.eig(train_Jacobian_2_layers)[0]))*train_Jacobian_2_layers)
    eig_vals=torch.linalg.eig(train_Jacobian_2_layers)[0]
    eigenvalues_train_2_layers = torch.max(torch.abs(torch.linalg.eig(train_Jacobian_2_layers)[0]))

    #print("largest eigenvalue: ", eigenvalues_train_2_layers)
   # print("largest eigenvalue_: ", eigenvalues_train_2_layers_)

    diff=torch.abs(eigenvalues_train_2_layers-eigenvalues_init_2_layers)
   # diff_=torch.abs(eigenvalues_train_2_layers_-eigenvalues_init_2_layers_)

   # print("difference Between largest eigenvalue norms at init. and after training : ",diff )
   # print("difference Between largest eigenvalue norms at init. and after training : ",diff )

    differences_2_layers.append(diff)
    return eig_vals,differences_2_layers
def calculate_jacobian_3_layers_tanh(init_weights,trained_weights,input):
    differences_3_layers=[]
    weights_init_layer_0 = init_weights["fc1.weight"]
    weights_init_layer_1 = init_weights["fc2.weight"]
    weights_init_layer_2 = init_weights["fc3.weight"]

    diag_2_init=make_diagonal_tanh(weights_init_layer_0,weights_init_layer_1,None,input,3)
    diag_1_init=make_diagonal_tanh(weights_init_layer_0,None,None,input,2)
    init_Jacobian_3_layers = 1/np.sqrt(1000)*weights_init_layer_2@diag_2_init@(1/np.sqrt(1000)*weights_init_layer_1)@diag_1_init@(1/np.sqrt(1000)*weights_init_layer_0)
    #print(init_Jacobian_3_layers)
    _, singular_values, _ = torch.svd(init_Jacobian_3_layers)
    operator_norm_init_2_layers = torch.max(singular_values)
   # print("Operator Norm of Initial Jacobian of 3 layers:", operator_norm_init_2_layers.item())
    eigenvalues_init_3_layers = torch.norm((torch.max(torch.abs(torch.linalg.eig(init_Jacobian_3_layers)[1])))*init_Jacobian_3_layers)
   # print("largest eigenvalue: ", eigenvalues_init_3_layers)
    ###trained Jacobian
    weights_train_layer_0 = trained_weights["fc1.weight"]
    weights_train_layer_1 = trained_weights["fc2.weight"]
    weights_train_layer_2 = trained_weights["fc3.weight"]
    diag_2_train = make_diagonal_tanh(weights_train_layer_0,weights_train_layer_1,None,input,3)
    diag_1_train = make_diagonal_tanh(weights_train_layer_0,None,None,input,2)
    train_Jacobian_3_layers = 1/np.sqrt(1000)*weights_train_layer_2@diag_2_train@(1/np.sqrt(1000)*weights_train_layer_1)@diag_1_train@(1/np.sqrt(1000)*weights_train_layer_0)
    _, singular_values_, _ = torch.svd(train_Jacobian_3_layers)
    operator_norm_train_3_layers = torch.max(singular_values_)
   # print("Operator Norm of Trained Jacobian of 3 layers:", operator_norm_train_3_layers.item())
    eigenvalues_train_3_layers =torch.norm( (torch.max(torch.abs(torch.linalg.eig(train_Jacobian_3_layers)[0])))*train_Jacobian_3_layers)
   # print("largest eigenvalue: ", eigenvalues_train_3_layers)
    diff=torch.abs(eigenvalues_train_3_layers-eigenvalues_init_3_layers)
  #  print("difference Between largest eigenvalue norms at init. and after training : ",diff )
    differences_3_layers.append(diff)
    return differences_3_layers
def calculate_jacobian_4_layers_tanh(init_weights,trained_weights,input):
    differences_4_layers=[]
    weights_init_layer_0 = init_weights["fc1.weight"]
    weights_init_layer_1 = init_weights["fc2.weight"]
    weights_init_layer_2 = init_weights["fc3.weight"]
    weights_init_layer_3 = init_weights["fc4.weight"]

    diag_1_init=make_diagonal_tanh(weights_init_layer_0,None,None,input,2)
    diag_2_init=make_diagonal_tanh(weights_init_layer_0,weights_init_layer_1,None,input,3)
    diag_3_init=make_diagonal_tanh(weights_init_layer_0,weights_init_layer_1,weights_init_layer_2,input,4)
    init_Jacobian_4_layers = 1/np.sqrt(1000)*weights_init_layer_3@diag_3_init@(1/np.sqrt(1000)*weights_init_layer_2)@diag_2_init@(1/np.sqrt(1000)*weights_init_layer_1)@diag_1_init@(1*np.sqrt(1000)*weights_init_layer_0)
    _, singular_values, _ = torch.svd(init_Jacobian_4_layers)
    operator_norm_init_2_layers = torch.max(singular_values)
   # print("Operator Norm of Initial Jacobian of 3 layers:", operator_norm_init_2_layers.item())
    eigenvalues_init_4_layers = torch.norm((torch.max(torch.abs(torch.linalg.eig(init_Jacobian_4_layers)[0])))*init_Jacobian_4_layers)
  #  print("largest eigenvalue: ", eigenvalues_init_4_layers)
    weights_train_layer_0 = trained_weights["fc1.weight"]
    weights_train_layer_1 = trained_weights["fc2.weight"]
    weights_train_layer_2 = trained_weights["fc3.weight"]
    weights_train_layer_3 = trained_weights["fc4.weight"]
    diag_1_train = make_diagonal_tanh(weights_train_layer_0,None,None,input,2)
    diag_2_train = make_diagonal_tanh(weights_train_layer_0,weights_train_layer_1,None,input,3)
    diag_3_train = make_diagonal_tanh(weights_train_layer_0,weights_train_layer_1,weights_train_layer_2,input,4)
    train_Jacobian_4_layers = 1/np.sqrt(1000)*weights_train_layer_3@diag_3_train@(1/np.sqrt(1000)*weights_train_layer_2)@diag_2_train@(1/np.sqrt(1000)*weights_train_layer_1)@diag_1_train@(1/np.sqrt(1000)*weights_train_layer_0)
    _, singular_values_, _ = torch.svd(train_Jacobian_4_layers)
    operator_norm_train_4_layers = torch.max(singular_values_)
   # print("Operator Norm of Trained Jacobian of 4 layers:", operator_norm_train_4_layers.item())
    eigenvalues_train_4_layers =torch.norm( (torch.max(torch.abs(torch.linalg.eig(train_Jacobian_4_layers)[1])))*train_Jacobian_4_layers)
  #  print("largest eigenvalue: ", eigenvalues_train_4_layers)
    diff=torch.abs(eigenvalues_train_4_layers-eigenvalues_init_4_layers)
   # print("difference Between largest eigenvalue norms at init. and after training : ",diff )
    differences_4_layers.append(diff)
    return differences_4_layers
def make_diagonal_erf(weights_1, weights_2, weights_3, input_, layers):
    if layers == 2:
            hidden_dim = weights_1.shape[1]

            temp = torch.mul(weights_1, 1 / np.sqrt(hidden_dim))
            w_alpha = torch.matmul(temp, input_.T)

            # sigmoid = torch.sigmoid(w_alpha)
            erf_derivative = 2 / torch.sqrt(torch.tensor(math.pi)) * torch.exp(-w_alpha**2)
            diag = torch.diagflat(erf_derivative)
            return diag
def calculate_jacobian_2_layers_erf(init_weights, trained_weights, input):
    differences_2_layers = []

    weights_init_layer_0 = init_weights["fc1.weight"]
    weights_init_layer_1 = init_weights["fc2.weight"]
    hidden_dim = init_weights["fc1.weight"].shape[0]
    input_dim = init_weights["fc1.weight"].shape[1]
    diag = make_diagonal_erf(weights_init_layer_0, None, None, input, 2)
    init_Jacobian_2_layers = 1 / np.sqrt(hidden_dim) * weights_init_layer_1 @ diag @ (1 / np.sqrt(input_dim) * weights_init_layer_0)
    # init_Jacobian_2_layers=1/4 *(1/np.sqrt(32)* weights_init_layer_1)@(weights_init_layer_0*1/np.sqrt(1000))

    _, singular_values, _ = torch.svd(init_Jacobian_2_layers)
    operator_norm_init_2_layers = torch.max(singular_values)
    # print("Operator Norm of Initial Jacobian of 2 layers:", operator_norm_init_2_layers.item())
    eigenvalues_init_2_layers_ = torch.norm(
        torch.max(torch.abs(torch.linalg.eig(init_Jacobian_2_layers)[0])) * init_Jacobian_2_layers)
    eigenvalues_init_2_layers = torch.max(torch.abs(torch.linalg.eig(init_Jacobian_2_layers)[0]))
    # print("largest eigenvalue_: ", eigenvalues_init_2_layers_)

    # print("largest eigenvalue: ", eigenvalues_init_2_layers)
    ###trained Jacobian
    weights_train_layer_0 = trained_weights["fc1.weight"]
    weights_train_layer_1 = trained_weights["fc2.weight"]
    diag_train = make_diagonal_erf(weights_train_layer_0, None, None, input, 2)
    train_Jacobian_2_layers = 1 / np.sqrt(hidden_dim) * weights_train_layer_1 @ diag_train @ (
                1 / np.sqrt(input_dim) * weights_train_layer_0)
    # train_Jacobian_2_layers=1/4 *1/np.sqrt(32)* weights_train_layer_1@weights_train_layer_0

    _, singular_values_, _ = torch.svd(train_Jacobian_2_layers)
    operator_norm_train_2_layers = torch.max(singular_values_)
    # print("Operator Norm of Trained Jacobian of 2 layers:", operator_norm_train_2_layers.item())
    eigenvalues_train_2_layers_ = torch.norm(
        torch.max(torch.abs(torch.linalg.eig(train_Jacobian_2_layers)[0])) * train_Jacobian_2_layers)
    eig_vals = torch.linalg.eig(train_Jacobian_2_layers)[0]
    eigenvalues_train_2_layers = torch.max(torch.abs(torch.linalg.eig(train_Jacobian_2_layers)[0]))

    # print("largest eigenvalue: ", eigenvalues_train_2_layers)
    # print("largest eigenvalue_: ", eigenvalues_train_2_layers_)

    diff = torch.abs(eigenvalues_train_2_layers - eigenvalues_init_2_layers)
    # diff_=torch.abs(eigenvalues_train_2_layers_-eigenvalues_init_2_layers_)

    # print("difference Between largest eigenvalue norms at init. and after training : ",diff )
    # print("difference Between largest eigenvalue norms at init. and after training : ",diff )

    differences_2_layers.append(diff)
    return eig_vals, differences_2_layers
def calculate_jacobian_2_layers_(init_weights,trained_weights,input):
    differences_2_layers=[]

    weights_init_layer_0 = init_weights["fc1.weight"]
    weights_init_layer_1 = init_weights["fc2.weight"]
    hidden_dim=init_weights["fc1.weight"].shape[0]
    input_dim=init_weights["fc1.weight"].shape[1]

    diag=make_diagonal_(weights_init_layer_0,None,None,input,2)
    init_Jacobian_2_layers =1/np.sqrt(hidden_dim)*weights_init_layer_1@diag@(1/np.sqrt(input_dim)*weights_init_layer_0)
    #init_Jacobian_2_layers=1/4 *(1/np.sqrt(32)* weights_init_layer_1)@(weights_init_layer_0*1/np.sqrt(1000))

    _, singular_values, _ = torch.svd(init_Jacobian_2_layers)
    operator_norm_init_2_layers = torch.max(singular_values)
    #print("Operator Norm of Initial Jacobian of 2 layers:", operator_norm_init_2_layers.item())
    eigenvalues_init_2_layers_ = torch.norm(torch.max(torch.abs(torch.linalg.eig(init_Jacobian_2_layers)[0]))*init_Jacobian_2_layers)
    eigenvalues_init_2_layers = torch.max(torch.abs(torch.linalg.eig(init_Jacobian_2_layers)[0]))
    #print("largest eigenvalue_: ", eigenvalues_init_2_layers_)

   # print("largest eigenvalue: ", eigenvalues_init_2_layers)
    ###trained Jacobian
    weights_train_layer_0 = trained_weights["fc1.weight"]
    weights_train_layer_1 = trained_weights["fc2.weight"]
    diag_train = make_diagonal_(weights_train_layer_0,None,None,input,2)
    train_Jacobian_2_layers = 1/np.sqrt(hidden_dim)*weights_train_layer_1@diag_train@(1/np.sqrt(input_dim)*weights_train_layer_0)
    #train_Jacobian_2_layers=1/4 *1/np.sqrt(32)* weights_train_layer_1@weights_train_layer_0

    _, singular_values_, _ = torch.svd(train_Jacobian_2_layers)
    operator_norm_train_2_layers = torch.max(singular_values_)
    #print("Operator Norm of Trained Jacobian of 2 layers:", operator_norm_train_2_layers.item())
    eigenvalues_train_2_layers_ = torch.norm(torch.max(torch.abs(torch.linalg.eig(train_Jacobian_2_layers)[0]))*train_Jacobian_2_layers)
    eig_vals=torch.linalg.eig(train_Jacobian_2_layers)[0]
    eigenvalues_train_2_layers = torch.max(torch.abs(torch.linalg.eig(train_Jacobian_2_layers)[0]))

    #print("largest eigenvalue: ", eigenvalues_train_2_layers)
   # print("largest eigenvalue_: ", eigenvalues_train_2_layers_)

    diff=torch.abs(eigenvalues_train_2_layers-eigenvalues_init_2_layers)
   # diff_=torch.abs(eigenvalues_train_2_layers_-eigenvalues_init_2_layers_)

   # print("difference Between largest eigenvalue norms at init. and after training : ",diff )
   # print("difference Between largest eigenvalue norms at init. and after training : ",diff )

    differences_2_layers.append(diff)
    return eig_vals,differences_2_layers



#3 layer

class Autoencoder_2_layers_relu(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder_2_layers_relu, self).__init__()
#super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.relu = nn.ReLU()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self._initialize_weights()

    def forward(self, x_):
        x_ = self.relu(1/np.sqrt( self.input_dim) * self.fc1(x_))
        x_ = 1/np.sqrt(self.hidden_dim) *self.fc2(x_)#32 for ex3 rest 1000
      #  x_ = self.sigmoid(x_)
        return x_

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
               # m.weight.data *= 1 / np.sqrt(m.weight.size(1))
class Autoencoder_3_layers_relu(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder_3_layers_relu, self).__init__()
#super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.relu = nn.ReLU()

        self._initialize_weights()

    def forward(self, x_):
        x_ = self.relu(1/np.sqrt(1000) * self.fc1(x_))
        x_ = 1/np.sqrt(1000) *self.fc2(x_)
        x_ = self.relu(x_)
        x_ = 1/np.sqrt(32) *self.fc3(x_)
        return x_

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
                m.weight.data *= 1 / np.sqrt(m.weight.size(1))
class Autoencoder_4_layers_relu(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder_4_layers_relu, self).__init__()
#super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc4 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.relu = nn.ReLU()

        self._initialize_weights()

    def forward(self, x_):
        x_ = self.relu(1/np.sqrt(1000) * self.fc1(x_))
        x_ = 1/np.sqrt(1000) *self.fc2(x_)
        x_ = self.relu(x_)
        x_ = 1/np.sqrt(1000) *self.fc3(x_)
        x_ = self.relu(x_)
        x_ = 1/np.sqrt(32) *self.fc4(x_)

        return x_

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
                m.weight.data *= 1 / np.sqrt(m.weight.size(1))

def relu_derivative(x):
    return 1. * (x > 0)

def make_diagonal_relu(weights_1, weights_2, weights_3, input_, layers):
    if layers == 2:
        hidden_dim = weights_1.shape[1]
        temp = torch.mul(weights_1, 1 / np.sqrt(hidden_dim))
        w_alpha = torch.matmul(temp, input_.T)
        relu_deriv = relu_derivative(w_alpha)
        diag = torch.diagflat(relu_deriv)
        return diag
    if layers == 3:
        temp = torch.mul(weights_1, 1 / np.sqrt(1000))
        w_alpha = torch.matmul(temp, input_.T)
        relu = torch.relu(w_alpha)
        relu_ = torch.matmul(torch.mul(1 / np.sqrt(1000), weights_2), relu)
        relu_deriv = relu_derivative(relu_)
        diag = torch.diagflat(relu_deriv)
        return diag
    if layers == 4:
        temp = torch.mul(weights_1, 1 / np.sqrt(1000))
        w_alpha = torch.matmul(temp, input_.T)
        relu = torch.relu(w_alpha)
        relu_ = torch.matmul(torch.mul(1 / np.sqrt(1000), weights_2), relu)
        relu_1 = torch.relu(relu_)
        relu_2 = torch.matmul(torch.mul(1 / np.sqrt(1000), weights_3), relu_1)
        relu_deriv = relu_derivative(relu_2)
        diag = torch.diagflat(relu_deriv)
        return diag
def calculate_jacobian_2_layers_relu(init_weights,trained_weights,input):
    differences_2_layers=[]

    weights_init_layer_0 = init_weights["fc1.weight"]
    weights_init_layer_1 = init_weights["fc2.weight"]
    hidden_dim=init_weights["fc1.weight"].shape[0]
    input_dim=init_weights["fc1.weight"].shape[1]
    diag=make_diagonal_relu(weights_init_layer_0,None,None,input,2)
    init_Jacobian_2_layers =1/np.sqrt(hidden_dim)*weights_init_layer_1@diag@(1/np.sqrt(hidden_dim)*weights_init_layer_0)
    #init_Jacobian_2_layers=1/4 *(1/np.sqrt(32)* weights_init_layer_1)@(weights_init_layer_0*1/np.sqrt(1000))

    _, singular_values, _ = torch.svd(init_Jacobian_2_layers)
    operator_norm_init_2_layers = torch.max(singular_values)
    #print("Operator Norm of Initial Jacobian of 2 layers:", operator_norm_init_2_layers.item())
    eigenvalues_init_2_layers_ = torch.norm(torch.max(torch.abs(torch.linalg.eig(init_Jacobian_2_layers)[0]))*init_Jacobian_2_layers)
    eigenvalues_init_2_layers = torch.max(torch.abs(torch.linalg.eig(init_Jacobian_2_layers)[0]))
    #print("largest eigenvalue_: ", eigenvalues_init_2_layers_)

   # print("largest eigenvalue: ", eigenvalues_init_2_layers)
    ###trained Jacobian
    weights_train_layer_0 = trained_weights["fc1.weight"]
    weights_train_layer_1 = trained_weights["fc2.weight"]
    diag_train = make_diagonal_relu(weights_train_layer_0,None,None,input,2)
    train_Jacobian_2_layers = 1/np.sqrt(hidden_dim)*weights_train_layer_1@diag_train@(1/np.sqrt(hidden_dim)*weights_train_layer_0)
    #train_Jacobian_2_layers=1/4 *1/np.sqrt(32)* weights_train_layer_1@weights_train_layer_0

    _, singular_values_, _ = torch.svd(train_Jacobian_2_layers)
    operator_norm_train_2_layers = torch.max(singular_values_)
    #print("Operator Norm of Trained Jacobian of 2 layers:", operator_norm_train_2_layers.item())
    eigenvalues_train_2_layers_ = torch.norm(torch.max(torch.abs(torch.linalg.eig(train_Jacobian_2_layers)[0]))*train_Jacobian_2_layers)
    eig_vals=torch.linalg.eig(train_Jacobian_2_layers)[0]
    eigenvalues_train_2_layers = torch.max(torch.abs(torch.linalg.eig(train_Jacobian_2_layers)[0]))

    #print("largest eigenvalue: ", eigenvalues_train_2_layers)
   # print("largest eigenvalue_: ", eigenvalues_train_2_layers_)

    diff=torch.abs(eigenvalues_train_2_layers-eigenvalues_init_2_layers)
   # diff_=torch.abs(eigenvalues_train_2_layers_-eigenvalues_init_2_layers_)

   # print("difference Between largest eigenvalue norms at init. and after training : ",diff )
   # print("difference Between largest eigenvalue norms at init. and after training : ",diff )

    differences_2_layers.append(diff)
    return eig_vals,differences_2_layers
def calculate_jacobian_3_layers_relu(init_weights,trained_weights,input):
    differences_3_layers=[]
    weights_init_layer_0 = init_weights["fc1.weight"]
    weights_init_layer_1 = init_weights["fc2.weight"]
    weights_init_layer_2 = init_weights["fc3.weight"]

    diag_2_init=make_diagonal_relu(weights_init_layer_0,weights_init_layer_1,None,input,3)
    diag_1_init=make_diagonal_relu(weights_init_layer_0,None,None,input,2)
    init_Jacobian_3_layers = 1/np.sqrt(1000)*weights_init_layer_2@diag_2_init@(1/np.sqrt(1000)*weights_init_layer_1)@diag_1_init@(1/np.sqrt(1000)*weights_init_layer_0)
    #print(init_Jacobian_3_layers)
    _, singular_values, _ = torch.svd(init_Jacobian_3_layers)
    operator_norm_init_2_layers = torch.max(singular_values)
   # print("Operator Norm of Initial Jacobian of 3 layers:", operator_norm_init_2_layers.item())
    eigenvalues_init_3_layers = torch.norm((torch.max(torch.abs(torch.linalg.eig(init_Jacobian_3_layers)[1])))*init_Jacobian_3_layers)
   # print("largest eigenvalue: ", eigenvalues_init_3_layers)
    ###trained Jacobian
    weights_train_layer_0 = trained_weights["fc1.weight"]
    weights_train_layer_1 = trained_weights["fc2.weight"]
    weights_train_layer_2 = trained_weights["fc3.weight"]
    diag_2_train = make_diagonal_relu(weights_train_layer_0,weights_train_layer_1,None,input,3)
    diag_1_train = make_diagonal_relu(weights_train_layer_0,None,None,input,2)
    train_Jacobian_3_layers = 1/np.sqrt(1000)*weights_train_layer_2@diag_2_train@(1/np.sqrt(1000)*weights_train_layer_1)@diag_1_train@(1/np.sqrt(1000)*weights_train_layer_0)
    _, singular_values_, _ = torch.svd(train_Jacobian_3_layers)
    operator_norm_train_3_layers = torch.max(singular_values_)
   # print("Operator Norm of Trained Jacobian of 3 layers:", operator_norm_train_3_layers.item())
    eigenvalues_train_3_layers =torch.norm( (torch.max(torch.abs(torch.linalg.eig(train_Jacobian_3_layers)[0])))*train_Jacobian_3_layers)
   # print("largest eigenvalue: ", eigenvalues_train_3_layers)
    diff=torch.abs(eigenvalues_train_3_layers-eigenvalues_init_3_layers)
  #  print("difference Between largest eigenvalue norms at init. and after training : ",diff )
    differences_3_layers.append(diff)
    return differences_3_layers
def calculate_jacobian_4_layers_relu(init_weights,trained_weights,input):
    differences_4_layers=[]
    weights_init_layer_0 = init_weights["fc1.weight"]
    weights_init_layer_1 = init_weights["fc2.weight"]
    weights_init_layer_2 = init_weights["fc3.weight"]
    weights_init_layer_3 = init_weights["fc4.weight"]

    diag_1_init=make_diagonal_relu(weights_init_layer_0,None,None,input,2)
    diag_2_init=make_diagonal_relu(weights_init_layer_0,weights_init_layer_1,None,input,3)
    diag_3_init=make_diagonal_relu(weights_init_layer_0,weights_init_layer_1,weights_init_layer_2,input,4)
    init_Jacobian_4_layers = 1/np.sqrt(1000)*weights_init_layer_3@diag_3_init@(1/np.sqrt(1000)*weights_init_layer_2)@diag_2_init@(1/np.sqrt(1000)*weights_init_layer_1)@diag_1_init@(1*np.sqrt(1000)*weights_init_layer_0)
    _, singular_values, _ = torch.svd(init_Jacobian_4_layers)
    operator_norm_init_2_layers = torch.max(singular_values)
   # print("Operator Norm of Initial Jacobian of 3 layers:", operator_norm_init_2_layers.item())
    eigenvalues_init_4_layers = torch.norm((torch.max(torch.abs(torch.linalg.eig(init_Jacobian_4_layers)[0])))*init_Jacobian_4_layers)
  #  print("largest eigenvalue: ", eigenvalues_init_4_layers)
    weights_train_layer_0 = trained_weights["fc1.weight"]
    weights_train_layer_1 = trained_weights["fc2.weight"]
    weights_train_layer_2 = trained_weights["fc3.weight"]
    weights_train_layer_3 = trained_weights["fc4.weight"]
    diag_1_train = make_diagonal_relu(weights_train_layer_0,None,None,input,2)
    diag_2_train = make_diagonal_relu(weights_train_layer_0,weights_train_layer_1,None,input,3)
    diag_3_train = make_diagonal_relu(weights_train_layer_0,weights_train_layer_1,weights_train_layer_2,input,4)
    train_Jacobian_4_layers = 1/np.sqrt(1000)*weights_train_layer_3@diag_3_train@(1/np.sqrt(1000)*weights_train_layer_2)@diag_2_train@(1/np.sqrt(1000)*weights_train_layer_1)@diag_1_train@(1/np.sqrt(1000)*weights_train_layer_0)
    _, singular_values_, _ = torch.svd(train_Jacobian_4_layers)
    operator_norm_train_4_layers = torch.max(singular_values_)
   # print("Operator Norm of Trained Jacobian of 4 layers:", operator_norm_train_4_layers.item())
    eigenvalues_train_4_layers =torch.norm( (torch.max(torch.abs(torch.linalg.eig(train_Jacobian_4_layers)[1])))*train_Jacobian_4_layers)
  #  print("largest eigenvalue: ", eigenvalues_train_4_layers)
    diff=torch.abs(eigenvalues_train_4_layers-eigenvalues_init_4_layers)
   # print("difference Between largest eigenvalue norms at init. and after training : ",diff )
    differences_4_layers.append(diff)
    return differences_4_layers