import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_dim):
        self.input_dim = input_dim 

        self.conv_1 = nn.Conv2D(input_dim, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.conv_2 = nn.Conv2D(input_dim, 128, kernel_size=5)
        self.conv_3 = nn.Conv2D(input_dim, 256, kernel_size=3)
        self.conv_4 = nn.Conv2D(input_dim, 512, kernel_size=3)
        self.conv_5 = nn.Conv2D(input_dim, 512, kernel_size=3)



        self.non_linearity = nn.ReLU()

        conv_3=Conv2D(256,(3,3),padding="same",activation='relu',name='conv_3')(conv_2)
        conv_3=MaxPooling2D(pool_size=(2, 2))(conv_3)
        conv_4=Conv2D(512,(3,3),padding="same",activation='relu',name='conv_4')(conv_3)
        conv_5=Conv2D(512,(3,3),padding="same",activation='relu',name='conv_5')(conv_4)
        conv_5=MaxPooling2D(pool_size=(2, 2))(conv_5)
    
        dense_1=Flatten()(conv_5)
        dense_1=Dense(512,activation="relu")(dense_1)
        dense_1=Dropout(0.5)(dense_1)
        dense_2=Dense(512,activation="relu")(dense_1)
        dense_2=Dropout(0.5)(dense_2)
        return Model(inputs, dense_2)

    def forward(self):
        pass