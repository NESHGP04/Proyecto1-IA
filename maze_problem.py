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
                if self.grid[ni][nj] is not None:
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
        i, j = next_state
        cell = self.grid[i][j]

        #si es pared
        if cell is None:
            return float("inf")

        #convertir a numpy y normalizar
        rgb = np.array(cell).reshape(1, -1) / 255.0

        prediction = self.model.predict(rgb)[0]

        label = self.idx_to_label[prediction]

        cost = self.label_to_cost.get(label, 1)

        return cost
