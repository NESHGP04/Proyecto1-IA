from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from search_problem import SearchProblem
from maze_problem import MazeProblem

'''
TASK 1.1
Para cada bloque:
Sacamos el promedio RGB
Si es:
    [0,0,0] → pared (rgba(0, 0, 0))
    Rojo dominante → inicio (rgba(254, 0, 0))
    Verde dominante → meta (rgba(5, 252, 6))
    Blanco → libre (rgba(255, 255, 255))
'''
#Leer imagen bmp y convertirla a RGB
def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

#Discretizar imagen
def discretize_image(img, tile_size=10):
    height, width, _ = img.shape
    
    grid_h = height // tile_size
    grid_w = width // tile_size
    
    grid = []
    start = None
    goals = []
    
    for i in range(grid_h):
        row = []
        for j in range(grid_w):
            
            tile = img[
                i*tile_size:(i+1)*tile_size,
                j*tile_size:(j+1)*tile_size
            ]
            
            avg_color = np.mean(tile.reshape(-1,3), axis=0)
            
            r, g, b = avg_color
            
            # Clasificación
            if np.all(avg_color < 20):   # negro
                row.append(1)  # pared
            elif r > 200 and g < 80 and b < 80:  # rojo
                row.append(2)  # inicio
                start = (i, j)
            elif g > 200 and r < 80 and b < 80:  # verde
                row.append(3)  # meta
                goals.append((i, j))
            else:
                row.append(0)  # libre
        
        grid.append(row)
    if start is None:
        raise ValueError("Imagen sin punto de inicio")

    if len(goals) == 0:
        raise ValueError("Imagen sin meta")

    return np.array(grid), start, goals

img = load_image("./Img/turing.bmp")
grid, start, goals = discretize_image(img, tile_size=10)

print("Start:", start)
print("Goals:", goals)
print("Grid shape:", grid.shape)

#Visualizar grid
plt.imshow(grid, cmap="gray")
plt.title("Grid Discretizado")
plt.show()

'''
Task 1.2 (BFS y DFS)
'''

def bfs(problem):
    start = problem.initial_state()
    
    frontier = deque([start])
    came_from = {start: None}
    
    while frontier:
        current = frontier.popleft()
        
        if problem.goal_test(current):
            return reconstruct_path(came_from, current)
        
        for action in problem.actions(current):
            next_state = problem.result(current, action)
            
            if next_state not in came_from:
                frontier.append(next_state)
                came_from[next_state] = current
    
    return None

def reconstruct_path(came_from, goal):
    path = []
    current = goal
    
    while current is not None:
        path.append(current)
        current = came_from[current]
    
    path.reverse()
    return path

def dfs(problem):
    start = problem.initial_state()
    
    stack = [start]
    came_from = {start: None}
    
    while stack:
        current = stack.pop()
        
        if problem.goal_test(current):
            return reconstruct_path(came_from, current)
        
        for action in problem.actions(current):
            next_state = problem.result(current, action)
            
            if next_state not in came_from:
                stack.append(next_state)
                came_from[next_state] = current
    
    return None

def visualize_path(grid, path):
    grid_copy = np.copy(grid)
    
    for (i, j) in path:
        if grid_copy[i][j] == 0:
            grid_copy[i][j] = 4  # marcar camino
    
    plt.imshow(grid_copy)
    plt.title("Path Found")
    plt.show()

problem = MazeProblem(grid, start, goals)

path_bfs = bfs(problem)
if path_bfs is not None:
    print("BFS Path length:", len(path_bfs))
    visualize_path(grid, path_bfs)
else:
    print("No se encontró camino con BFS")


path_dfs = dfs(problem)
if path_dfs is not None:
    print("DFS Path length:", len(path_dfs))
    visualize_path(grid, path_dfs)
else:
    print("No se encontró camino con DFS")

