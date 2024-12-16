import numpy as np
import random

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

class Board:
    def __init__(self, size):
        self.size = size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.get_state()

    def get_state(self):
        return self.board.flatten()

    def get_empty_cells(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i, j] == 0]

    def add_random_tile(self):
        empty_cells = self.get_empty_cells()
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.board[r, c] = 2 if random.random() > 0.9 else 2

    def is_over(self):
        if np.any(self.board == 0):
            return False  
        
        for r in range(self.size):
            for c in range(self.size):
                if c + 1 < self.size and self.board[r, c] == self.board[r, c + 1]:
                    return False
                if r + 1 < self.size and self.board[r, c] == self.board[r + 1, c]:
                    return False
        return True  
    
    def move_left(self):
        moved = False
        for row in range(self.size):
            non_zero = [self.board[row, col] for col in range(self.size) if self.board[row, col] != 0]
            new_row = self.merge_(non_zero)
            for col in range(self.size):
                if self.board[row, col] != new_row[col]:
                    moved = True
                self.board[row, col] = new_row[col]
        return moved
    
    def step(self, direct):
        old_score = self.score
        self.board = np.rot90(self.board, direct)
        moved = self.move_left()
        self.board = np.rot90(self.board, -direct)

        if moved:
            self.add_random_tile()
            
        return self.get_state(), self.score - old_score, self.is_over()

    def merge_(self, non_zero):
        merged = []
        skip = False
        for i in range(len(non_zero)):
            if skip:
                skip = False
            elif i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged.append(non_zero[i] * 2)
                self.score += non_zero[i] * 2
                skip = True
            else:
                merged.append(non_zero[i])
        merged += [0] * (self.size - len(merged)) 
        return merged


env = Board(size=4)

import math
import torch
import torch.nn.functional as F

def encode_board(board : np.array):
    state = torch.tensor([0 if value == 0 else int(math.log(value, 2)) for value in board])
    state = F.one_hot(state, num_classes=16).float().flatten()
    state = state.reshape(1, 4, 4, 16).permute(0, 3, 1, 2)
    return state