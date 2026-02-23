from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from search_problem import SearchProblem
from maze_problem import MazeProblem
import heapq

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
            
            pixels = tile.reshape(-1, 3)
            total_pixels = len(pixels)

            # ----------------------------
            # DETECCIÓN DE PARED (NEGRO)
            # ----------------------------
            black_pixels = np.sum(
                (pixels[:,0] < 50) &
                (pixels[:,1] < 50) &
                (pixels[:,2] < 50)
            )
            black_ratio = black_pixels / total_pixels

            # ----------------------------
            # DETECCIÓN DE INICIO (ROJO)
            # ----------------------------
            red_pixels = np.sum(
                (pixels[:,0] > 200) &
                (pixels[:,1] < 80) &
                (pixels[:,2] < 80)
            )
            red_ratio = red_pixels / total_pixels

            # ----------------------------
            # DETECCIÓN DE META (VERDE)
            # ----------------------------
            green_pixels = np.sum(
                (pixels[:,1] > 200) &
                (pixels[:,0] < 80) &
                (pixels[:,2] < 80)
            )
            green_ratio = green_pixels / total_pixels

            # ----------------------------
            # CLASIFICACIÓN FINAL
            # ----------------------------
            avg_color = np.mean(pixels, axis=0)
            avg_color = tuple(avg_color.astype(int))

            if black_ratio > 0.52:
                row.append(None)

            elif red_ratio > 0.4:
                row.append(avg_color)
                start = (i, j)

            elif green_ratio > 0.4:
                row.append(avg_color)
                goals.append((i, j))

            else:
                row.append(avg_color)
        
        grid.append(row)

    if start is None:
        raise ValueError("Imagen sin punto de inicio")

    if len(goals) == 0:
        raise ValueError("Imagen sin meta")

    return grid, start, goals

img = load_image("./Img/turing.bmp")
grid, start, goals = discretize_image(img, tile_size=10)

print("Start:", start)
print("Goals:", goals)
print("Grid shape:", (len(grid), len(grid[0])))

#unique, counts = np.unique(grid, return_counts=True)
#print(dict(zip(unique, counts)))


#Visualizar grid
def visualize_grid(grid):
    img = np.zeros((len(grid), len(grid[0]), 3), dtype=np.uint8)

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] is None:
                img[i][j] = [0, 0, 0]  # pared negra
            else:
                img[i][j] = grid[i][j]  # color real

    plt.imshow(img)
    plt.title("Grid Discretizado")
    plt.show()


visualize_grid(grid)

'''
Task 1.2 (BFS y DFS)
'''

def bfs(problem):
    start = problem.initial_state()
    
    frontier = deque([start])
    came_from = {start: None}
    steps = 0

    while frontier:
        current = frontier.popleft()
        steps += 1
        
        if steps % 500 == 0:
            print("Explorados:", steps)

        if problem.goal_test(current):
            print("Meta encontrada en", steps, "expansiones")
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

    height = len(grid)
    width = len(grid[0])

    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Convertir grid a imagen RGB
    for i in range(height):
        for j in range(width):
            if grid[i][j] is None:
                img[i][j] = [0, 0, 0]  # pared
            else:
                img[i][j] = grid[i][j]  # color real

    # Dibujar camino en morado
    for (i, j) in path:
        img[i][j] = [255, 0, 255]

    plt.imshow(img)
    plt.title("Path Found")
    plt.show()

print("Creando problema...")
problem = MazeProblem(grid, start, goals)
print("Problema creado.")

print("Ejecutando BFS...")
path_bfs = bfs(problem)
print("BFS terminó")

print("Resultado BFS:", path_bfs)

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

'''
Task 1.3 (A*) 
'''
#Heurística
def manhattan(state, goals):
    return min(
        abs(state[0] - goal[0]) + abs(state[1] - goal[1])
        for goal in goals
    )

#A*
def astar(problem):
    
    start = problem.initial_state()
    
    frontier = []
    heapq.heappush(frontier, (0, start))
    
    came_from = {start: None}
    g_cost = {start: 0}
    
    while frontier:
        _, current = heapq.heappop(frontier)
        
        if problem.goal_test(current):
            return reconstruct_path(came_from, current)
        
        for action in problem.actions(current):
            neighbor = problem.result(current, action)
            
            tentative_g = g_cost[current] + problem.step_cost(current, action, neighbor)
            
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                
                f = tentative_g + manhattan(neighbor, problem.goals)
                
                heapq.heappush(frontier, (f, neighbor))
                came_from[neighbor] = current
    
    return None

path_astar = astar(problem)

if path_astar is not None:
    print("A* Path length:", len(path_astar))
    visualize_path(grid, path_astar)
else:
    print("No se encontró camino con A*")
