# Template file to create an AI for the game PyRat
# http://formations.telecom-bretagne.eu/pyrat

###############################
# Team name to be displayed in the game 
TEAM_NAME = "Q-Learner"

###############################
# When the player is performing a move, it actually sends a character to the main program
# The four possibilities are defined here
MOVE_DOWN = 'D'
MOVE_LEFT = 'L'
MOVE_RIGHT = 'R'
MOVE_UP = 'U'

###############################
# Please put your imports here

import numpy as np
import random as rd
import pickle
import time
import torch
import torch.nn as nn

###############################
# Please put your global variables here

# Global variables
global model, exp_replay, input_tm1, action, score

# Function to create a numpy array representation of the maze

def input_of_parameters(player, maze, opponent, mazeHeight, mazeWidth, piecesOfCheese):
    im_size = (2 * mazeHeight - 1, 2 * mazeWidth - 1, 2)
    canvas = np.zeros(im_size)
    (x,y) = player
    (x_enemy, y_enemy) = opponent
    center_x, center_y = mazeWidth-1, mazeHeight-1
    for (x_cheese,y_cheese) in piecesOfCheese:
        canvas[y_cheese + center_y - y, x_cheese + center_x - x, 0] = 1
        #self.canvas[y_cheese+center_y-y_enemy,x_cheese+center_x-x_enemy,1] = 1
    canvas[y_enemy+center_y-y,x_enemy+center_x-x,1] = 1    
#   (x_enemy, y_enemy) = opponent
#   canvas[y_enemy+center_y-y,x_enemy+center_x-x,1] = 1
#   canvas[center_y,center_x,2] = 1
    canvas = np.expand_dims(canvas, axis=0)
    return canvas


class MultiRegressorCNN3FC(nn.Module):
    def __init__(self, number_of_channels=1):
        super().__init__()
        self.nb_channels = number_of_channels
        self.conv1 = nn.Conv2d(number_of_channels, 16, kernel_size=3) # output_shape = (1, 16, 27, 39)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) # output_shape = (1, 16, 13, 19)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3) # output_shape = (1, 32, 11, 17)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) # output_shape = (1, 32, 5, 8)
        self.fc1 = nn.Linear(32 * 5 * 8, 10)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(10, 4)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        #x = self.dropout(x)
        
        x = x.reshape(x.shape[0],-1) # output_shape = (1280)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def load(self):
        if self.nb_channels == 1:
            self.load_state_dict(torch.load('save_rl/weights_CNN3FC_1channel.pt'))
        else:
            self.load_state_dict(torch.load('D:\PyRat-1\AIs\save_rl\weights_CNN3FC_2channel.pt'))

    def save(self):
        if self.nb_channels == 1:
            torch.save(self.state_dict(), 'save_rl/weights_CNN3FC_1channel.pt')
        else:
            torch.save(self.state_dict(), 'D:\PyRat-1\AIs\save_rl\weights_CNN3FC_2channel.pt')

    
###############################
# Preprocessing function
# The preprocessing function is called at the start of a game
# It can be used to perform intensive computations that can be
# used later to move the player in the maze.
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int,int)
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is not expected to return anything
def preprocessing(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, piecesOfCheese, timeAllowed):
    global model,exp_replay,input_tm1, action, score
    input_tm1 = input_of_parameters(playerLocation, mazeMap, opponentLocation, mazeHeight, mazeWidth, piecesOfCheese)    
    action = -1
    score = 0
    #model = MultiRegressorCNN3FC(input_tm1[0])
    model = MultiRegressorCNN3FC(input_tm1.shape[3])
    model.load()
    
###############################
# Turn function
# The turn function is called each time the game is waiting
# for the player to make a decision (a move).
###############################
# Arguments are:
# mazeMap : dict(pair(int, int), dict(pair(int, int), int))
# mazeWidth : int
# mazeHeight : int
# playerLocation : pair(int, int)
# opponentLocation : pair(int, int)
# playerScore : float
# opponentScore : float
# piecesOfCheese : list(pair(int, int))
# timeAllowed : float
###############################
# This function is expected to return a move
def turn(mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):    
    global model,input_tm1, action, score
    input_t = input_of_parameters(playerLocation, mazeMap, opponentLocation, mazeHeight, mazeWidth, piecesOfCheese)    
    input_tm1 = torch.FloatTensor(input_t)   
    #output = model(input_tm1.unsqueeze(dim=0))
    output = model(input_tm1)
    action = torch.argmax(output[0]).item()
    score = playerScore
    return [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN][action]

def postprocessing (mazeMap, mazeWidth, mazeHeight, playerLocation, opponentLocation, playerScore, opponentScore, piecesOfCheese, timeAllowed):
    pass    
