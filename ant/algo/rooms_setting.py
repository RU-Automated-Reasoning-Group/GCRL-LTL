import numpy as np


# AbstarctState class representing a rectangular region
class AbstractState:

    ## region: [(x1,y1),(x2,y2)]
    def __init__(self, region):
        self.region = np.array(region)
        self.size = self.region[1] - self.region[0]

    # s: np.array(2) or array-like
    def contains(self, s):
        return s[0] >= self.region[0][0] and s[0] <= self.region[1][0] \
            and s[1] >= self.region[0][1] and s[1] <= self.region[1][1]

    # sample a point from the region
    def sample(self):
        return np.random.random_sample(2) * self.size + self.region[0]

        
# parameters for defining the rooms environment
class GridParams:
    # size: (h:int, w:int) specifying size of grid
    # edges: list of pairs of adjacent rooms (room is a pair (x,y) - 0 based indexing)
    #        first coordinate is the vertical position (just like matrix indexing)
    # room_size: (l:int, b:int) size of a single room (height first)
    # wall_size: (tx:int, ty:int) thickness of walls (thickness of horizontal wall first)
    # vertical_door, horizontal_door: relative coordinates for door, specifies min and max
    #                                 coordinates for door space
    def __init__(self, size, edges, room_size, wall_size, vertical_door, horizontal_door):
        self.size = np.array(size)
        self.edges = edges
        self.room_size = np.array(room_size)
        self.wall_size = np.array(wall_size)
        self.partition_size = self.room_size + self.wall_size
        self.vdoor = np.array(vertical_door)
        self.hdoor = np.array(horizontal_door)
        self.graph = self.make_adjacency_matrix()
        self.grid_region = AbstractState([np.array([0., 0.]), self.size * self.partition_size])

    # map a room to an integer
    def get_index(self, r):
        return self.size[1]*r[0] + r[1]

    # returns the direction of r2 from r1
    def get_direction(self, r1, r2):
        if r1[0] == r2[0]+1 and r1[1] == r2[1]:
            return 0  # up
        elif r1[0] == r2[0] and r1[1] == r2[1]+1:
            return 1  # left
        elif r1[0] == r2[0]-1 and r1[1] == r2[1]:
            return 2  # down
        elif r1[0] == r2[0] and r1[1] == r2[1]-1:
            return 3  # right
        else:
            raise Exception('Given rooms are not adjacent!')

    # takes pairs of adjacent rooms and creates a h*w-by-4 matrix of booleans
    # returns the compact adjacency matrix
    def make_adjacency_matrix(self):
        graph = [[False]*4 for _ in range(self.size[0]*self.size[1])]
        for r1, r2 in self.edges:
            graph[self.get_index(r1)][self.get_direction(r1, r2)] = True
            graph[self.get_index(r2)][self.get_direction(r2, r1)] = True
        return graph

    # Region corresponding to the center of a room
    def get_center_region(self, room):
        center = self.partition_size * np.array(room) + (self.room_size / 2)
        half_size = self.wall_size / 2
        return AbstractState([center - half_size, center + half_size])

    # get predicate corresponding to center of room
    def in_room(self, room):
        center = self.partition_size * np.array(room) + (self.room_size / 2)
        half_size = self.wall_size / 2
        low = center - half_size
        high = center + half_size

        def predicate(sys_state, res_state):
            return min(np.concatenate([sys_state[:2] - low, high - sys_state[:2]]))

        return predicate

    # get predicate to avoid the center of a room
    def avoid_center(self, room):
        center = self.partition_size * np.array(room) + (self.room_size / 2)
        half_size = self.wall_size / 2
        low = center - half_size
        high = center + half_size

        def predicate(sys_state, res_state):
            return 10*max(np.concatenate(low - [sys_state[:2], sys_state[:2] - high]))

        return predicate



GRID_PARAMS_LIST = []
MAX_TIMESTEPS = []
START_ROOM = []
FINAL_ROOM = []

# parameters for a 3-by-3 grid
size1 = (3, 3)
edges1 = [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((0, 1), (0, 2)),
          ((1, 0), (1, 1)), ((1, 0), (2, 0)), ((1, 1), (1, 2)),
          ((2, 0), (2, 1)), ((2, 2), (1, 2))]
room_size1 = (8, 8)
wall_size1 = (2, 2)
vertical_door1 = (3, 5)
horizontal_door1 = (3, 5)

GRID_PARAMS_LIST.append(GridParams(size1, edges1, room_size1, wall_size1,
                                   vertical_door1, horizontal_door1))
MAX_TIMESTEPS.append(150)
START_ROOM.append((0, 0))
FINAL_ROOM.append((2, 2))

# parameters for a 2-by-2 grid
size2 = (2, 2)
edges2 = [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((1, 0), (1, 1))]
room_size2 = (8, 8)
wall_size2 = (2, 2)
vertical_door2 = (3, 5)
horizontal_door2 = (3, 5)

GRID_PARAMS_LIST.append(GridParams(size2, edges2, room_size2, wall_size2,
                                   vertical_door2, horizontal_door2))
MAX_TIMESTEPS.append(100)
START_ROOM.append((0, 0))
FINAL_ROOM.append((1, 1))

# parameters for a 3-by-3 grid
edges3 = [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((0, 1), (0, 2)),
          ((1, 0), (1, 1)), ((1, 0), (2, 0)), ((1, 1), (1, 2)),
          ((2, 0), (2, 1)), ((2, 1), (2, 2)), ((2, 2), (1, 2))]

GRID_PARAMS_LIST.append(GridParams(size1, edges3, room_size1, wall_size1,
                                   vertical_door1, horizontal_door1))
MAX_TIMESTEPS.append(150)
START_ROOM.append((0, 0))
FINAL_ROOM.append((2, 2))

# parameters for a 4-by-4 grid
size4 = (4, 4)
edges4 = []
non_edges4 = {((2, 3), (3, 3))}
for x in range(4):
    for y in range(4):
        down_edge = ((x, y), (x+1, y))
        right_edge = ((x, y), (x, y+1))
        if x+1 < 4 and down_edge not in non_edges4:
            edges4.append(down_edge)
        if y+1 < 4 and right_edge not in non_edges4:
            edges4.append(right_edge)
edges4.append(((2, 3), (3, 3)))

GRID_PARAMS_LIST.append(GridParams(size4, edges4, room_size1, wall_size1,
                                   vertical_door1, horizontal_door1))
MAX_TIMESTEPS.append(180)
START_ROOM.append((0, 0))
FINAL_ROOM.append((3, 3))

# parameters for a 4-by-4 grid
size5 = (4, 4)
edges5 = []
non_edges5 = {((2, 3), (3, 3)), ((0, 1), (1, 1)), ((0, 2), (1, 2)),
              ((1, 3), (2, 3)), ((2, 2), (3, 2))}
for x in range(4):
    for y in range(4):
        down_edge = ((x, y), (x+1, y))
        right_edge = ((x, y), (x, y+1))
        if x+1 < 4 and down_edge not in non_edges5:
            edges5.append(down_edge)
        if y+1 < 4 and right_edge not in non_edges5:
            edges5.append(right_edge)
edges5.append(((2, 3), (3, 3)))

GRID_PARAMS_LIST.append(GridParams(size5, edges5, room_size1, wall_size1,
                                   vertical_door1, horizontal_door1))
MAX_TIMESTEPS.append(180)
START_ROOM.append((0, 0))
FINAL_ROOM.append((3, 3))


