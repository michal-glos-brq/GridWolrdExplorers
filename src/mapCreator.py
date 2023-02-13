import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import json
import datetime
import os
  
EMPTY = 0
OBSTACLE = 2
AGENT = 4

class CreativeBoard():
    def __init__(self, save_path, dims=(10,10)) -> None:
        self.save_path = (save_path if save_path.endswith('.json') else save_path + ".json") if save_path else None
        self.poso = np.empty((0,2), dtype=int)
        self.posa = np.empty((0,2), dtype=int)
        self.gt_board = np.zeros(dims)
        self.dims = dims

        self.cmap = mpl.colors.ListedColormap(['honeydew', 'fuchsia', 'navy'])
        bounds=[-1,1,3,5]
        self.norm = mpl.colors.BoundaryNorm(bounds, self.cmap.N)
        
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(self.gt_board, interpolation='none', cmap=self.cmap,norm=self.norm)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)     
        self.fig.canvas.mpl_connect('key_press_event', self.onclick)     

        plt.axis('off')
        plt.show()

    def save_map(self):
        if not os.path.exists('./saved_maps'):
            os.makedirs('./saved_maps')
        data = {
            "posa": self.posa.tolist(),
            "poso": self.poso.tolist() 
        }
        if not self.save_path:
            self.save_path = f"a{len(data['posa'])}_o{len(data['poso'])}_{str(self.dims[0])}x{str(self.dims[1])}.json"
        with open(os.path.join('saved_maps', self.save_path), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def update(self):
        self.img = self.ax.imshow(self.gt_board, interpolation='none', cmap=self.cmap,norm=self.norm)
        self.fig.canvas.draw_idle()

    def onclick(self, event):
        if event.key == 'enter':
            self.poso = np.array(np.where(self.gt_board == OBSTACLE)).transpose() 
            self.posa = np.array(np.where(self.gt_board == AGENT)).transpose() 
            plt.close()
        elif event.key == 'j':
            self.poso = np.array(np.where(self.gt_board == OBSTACLE)).transpose() 
            self.posa = np.array(np.where(self.gt_board == AGENT)).transpose()
            self.save_map()
            plt.close()
        else:
            if event.ydata and event.xdata:
                coordinates = [int(round(event.ydata)), int(round(event.xdata))]
                if  event.button == 1:
                    if self.gt_board[coordinates[0], coordinates[1]] == EMPTY:
                        self.gt_board[coordinates[0], coordinates[1]] = OBSTACLE
                        self.update()
                    elif self.gt_board[coordinates[0], coordinates[1]] == OBSTACLE:
                        self.gt_board[coordinates[0], coordinates[1]] = AGENT
                        self.update()
                    else:
                        self.gt_board[coordinates[0], coordinates[1]] = EMPTY
                        self.update()