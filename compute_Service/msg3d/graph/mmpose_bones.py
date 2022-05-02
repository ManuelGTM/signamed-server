import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools

num_node = 51
self_link = [(i, i) for i in range(num_node)]

inward = [(0, 0),(0, 1),(0, 2),(0, 3),(0, 4),(3, 5),(4, 6),(5, 7),(6, 8),
          (9, 9),(9, 10),(10, 11),(11, 12),(12, 13),(9, 14),(14, 15),(15, 16),(16, 17),(9, 18),(18, 19),(19, 20),(20, 21),(9, 22),(22, 23),(23, 24),(24, 25),(9, 26),(26, 27),(27, 28),(28, 29),
          (30, 30),(30, 31),(31, 32),(32, 33),(33, 34),(30, 35),(35, 36),(36, 37),(37, 38),(30, 39),(39, 40),(40, 31),(41, 42),(30, 43),(43, 44),(44, 45),(45, 46),(30, 47),(47, 48),(48, 49),(49, 50)
          ]

outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)


if __name__ == '__main__':
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    import matplotlib.pyplot as plt
    print(A_binary)
    plt.matshow(A_binary)
    plt.show()
