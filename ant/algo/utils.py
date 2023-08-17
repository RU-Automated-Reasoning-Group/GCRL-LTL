import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import collections as mc
from collections import deque


class Line():
    def __init__(self, p0, p1):
        self.p0 = np.array(p0, dtype=float)
        self.p1 = np.array(p1, dtype=float)
        self.dirn = np.array(p1, dtype=float) - np.array(p0, dtype=float)
        self.dist = np.linalg.norm(self.dirn)
        self.dirn /= self.dist  # normalize

    def get_point(self, t):
        return self.p0 + t * self.dirn


class Box_obstacle():

    def __init__(self, top_left, top_right, down_left, down_right, collision_distance):
        self.TL = np.array([top_left[0] - collision_distance, top_left[1] + collision_distance], dtype=float)
        self.TR = np.array([top_right[0] + collision_distance, top_right[1] + collision_distance], dtype=float)
        self.DL = np.array([down_left[0] - collision_distance, down_left[1] - collision_distance], dtype=float)
        self.DR = np.array([down_right[0] + collision_distance, down_right[1] - collision_distance], dtype=float)

        self.edges_line = []
        self.edges_line.append([self.TL, self.TR])
        self.edges_line.append([self.TL, self.DL])
        self.edges_line.append([self.DR, self.TR])
        self.edges_line.append([self.DR, self.DL])

    def intersect_with_box(self, line):
        # '''There can not be such case that the whole line segment is within box obstacle in any ant envs'''
        line_seg = [line.p0, line.p1]
        p1 = line_seg[0]
        p2 = line_seg[1]
        for egde_line in self.edges_line:
            p3 = egde_line[0]
            p4 = egde_line[1]
            cross_vec_1 = np.cross(p1-p3, p4-p3)
            cross_vec_2 = np.cross(p2-p3, p4-p3)
            cross_vec_3 = np.cross(p3-p2, p1-p2)
            cross_vec_4 = np.cross(p4-p2, p1-p2)
            if np.dot(cross_vec_1, cross_vec_2) <= 0 and np.dot(cross_vec_3, cross_vec_4) <= 0:
                return True
        return False

    def within_box(self, point):
        # check if 'point' is covered by the region of box
        max_x = self.TR[0]
        max_y = self.TR[1]
        min_x = self.DL[0]
        min_y = self.DL[1]
        return (min_x < point[0] < max_x) and (min_y < point[1] < max_y)


def distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))

# def Intersection(line, center, radius):
#     ''' Check line-sphere (circle) intersection '''

#     ###
#     # line intersect with circle
#     # line: p' = p + dirn * t
#     # circle: (p'.x - obs.x)^2 + (p'.y - obs.y)^2 = r^2
#     # => (dirn.x^2 + dirn.y^2) * t^2 + 2(dirnx * (p.x - obs.x) + dirn.y * (p.y - obs.y)) * t + (p.x^2 + p.y^2 - r^2) = 0
#     # b^2 - 4 * a * c >? 0
#     ###
#     a = np.dot(line.dirn, line.dirn)
#     b = 2 * np.dot(line.dirn, line.p - center)
#     c = np.dot(line.p - center, line.p - center) - radius * radius

#     discriminant = b * b - 4 * a * c
#     if discriminant < 0:
#         return False

#     t1 = (-b + np.sqrt(discriminant)) / (2 * a)
#     t2 = (-b - np.sqrt(discriminant)) / (2 * a)

#     # restrict the line to a segment
#     if (t1 < 0 and t2 < 0) or (t1 > line.dist and t2 > line.dist):
#         return False

#     return True


def Draw_GridWorld(ax, maze_structure, maze_scaling):
    """
    For: AntU, AntFall, AntPush, AntFourrooms, AntLongU, AntS, AntPi, AntOmega
    """
    # Draw boundary
    len_y = len(maze_structure)
    len_x = len(maze_structure[0])

    # ax.hlines([-0.5 * maze_scaling], -1 * maze_scaling, len_x * maze_scaling, colors='b', linestyles='dashed')
    # ax.vlines([-0.5 * maze_scaling], -1 * maze_scaling, len_y * maze_scaling, colors='b', linestyles='dashed')
    # for i in range(len_y):
    #     ax.hlines([(i + 0.5) * maze_scaling], -1 * maze_scaling, len_x * maze_scaling, colors='b', linestyles='dashed')
    # for i in range(len_x):
    #     ax.vlines([(i + 0.5) * maze_scaling], -1 * maze_scaling, len_y * maze_scaling, colors='b', linestyles='dashed')

    # Draw obstacles
    for j in range(len_y):
        for i in range(len_x):
            if maze_structure[j][i] == 1:
                rect = patches.Rectangle(((i - 0.5) * maze_scaling, (j - 0.5) * maze_scaling),
                                         maze_scaling, maze_scaling, linewidth=3, edgecolor='k', facecolor='k')
                ax.add_patch(rect)

    ax.set_xlim(-1 * maze_scaling, len_x * maze_scaling)
    ax.set_ylim(-1 * maze_scaling, len_y * maze_scaling)
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(1 * maze_scaling))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1 * maze_scaling))
    ax.set_aspect(1)

    ax.axis('off')
    return ax

