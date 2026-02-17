from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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
    
    return np.array(grid), start, goals
