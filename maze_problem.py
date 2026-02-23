from search_problem import SearchProblem
import numpy as np
from mlp_model import MLP

class MazeProblem(SearchProblem):
    
    def __init__(self, grid, start, goals):
        self.grid = grid
        self.start = start
        self.goals = goals
        self.rows = len(grid)
        self.cols = len(grid[0])
    
    def initial_state(self):
        return self.start
    
    def goal_test(self, state):
        return state in self.goals
    
    def actions(self, state):
        i, j = state
        moves = []
        
        possible = {
            "UP": (i-1, j),
            "DOWN": (i+1, j),
            "LEFT": (i, j-1),
            "RIGHT": (i, j+1)
        }
        
        for action, (ni, nj) in possible.items():
            if 0 <= ni < self.rows and 0 <= nj < self.cols:
                if self.grid[ni][nj] != 1:  # no pared
                    moves.append(action)
        
        return moves
    
    def result(self, state, action):
        i, j = state
        
        if action == "UP":
            return (i-1, j)
        elif action == "DOWN":
            return (i+1, j)
        elif action == "LEFT":
            return (i, j-1)
        elif action == "RIGHT":
            return (i, j+1)
    
    def step_cost(self, state, action, next_state):
        return 1
