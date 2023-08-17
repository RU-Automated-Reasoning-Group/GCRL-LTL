import numpy as np
import gym
import math


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
            return [room, min(np.concatenate([sys_state[:2] - low, high - sys_state[:2]]))]

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


# Environment modelling 2d grid with rooms
class RoomsEnv(gym.Env):

    # grid_params: GridParams
    # start_room: tuple (x, y)
    # goal_room: tuple (x, y)
    def __init__(self, grid_params, start_room, goal_room, max_timesteps=1000):
        self.grid_params = grid_params
        self.start_region = self.grid_params.get_center_region(start_room)
        self.goal_region = self.grid_params.get_center_region(goal_room)
        self.max_timesteps = max_timesteps

        max_vel = np.amin(self.grid_params.wall_size) / 2
        self.action_scale = np.array([max_vel, np.pi/2])

        # set the initial state
        self.reset()

    def reset(self):
        self.steps = 0
        self.state = self.start_region.sample()
        return self.state

    def step(self, action):
        action = self.action_scale * action
        action = np.array([action[0] * math.cos(action[1]),
                           action[0] * math.sin(action[1])])
        next_state = self.state + action
        if self.path_clear(self.state, next_state):
            self.state = next_state
            self.steps += 1
            reward = 0
            done = self.steps > self.max_timesteps
            if self.goal_region.contains(next_state):
                reward = 1
                done = True
            return self.state, reward, done, {}
        else:
            reward = 0
            done = True
            return self.state, reward, done, {}

    @property
    def observation_space(self):
        shape = self.state.shape
        high = np.inf * np.ones(shape)
        low = -high
        return gym.spaces.Box(low, high, dtype=np.float32)

    @property
    def action_space(self):
        high = np.array([1., 1.])
        low = -high
        return gym.spaces.Box(low, high, dtype=np.float32)

    def render(self):
        pass

    def get_sim_state(self):
        return self.state

    def set_sim_state(self, state):
        self.state = state
        return self.state

    def close(self):
        pass

    # Check if straight line joining s1 and s2 does not pass through walls
    # s1 is assumed to be a legal state
    # we are assuming that wall size exceeds maximum action size
    # also assuming that door regions are small compared to rooms
    def path_clear(self, s1, s2):

        params = self.grid_params

        # find rooms of the states
        r1 = (s1//params.partition_size).astype(np.int)
        r2 = (s2//params.partition_size).astype(np.int)

        # find relative positions within rooms
        p1 = s1 - (r1 * params.partition_size)
        p2 = s2 - (r2 * params.partition_size)

        if not self.is_state_legal(s2, r2, p2):
            return False

        # both states are inside the same room (not in the door area)
        if (p1[0] <= params.room_size[0] and p1[1] <= params.room_size[1]
                and p2[0] <= params.room_size[0] and p2[1] <= params.room_size[1]):
            return True
        # both states in door area
        if ((p1[0] > params.room_size[0] or p1[1] > params.room_size[1])
                and (p2[0] > params.room_size[0] or p2[1] > params.room_size[1])):
            return True

        # swap to make sure s1 is in the room and s2 is in the door area
        if (p2[0] <= params.room_size[0] and p2[1] <= params.room_size[1]):
            p1, p2 = p2, p1
            r1, r2 = r2, r1
            s1, s2 = s2, s1

        # four cases to consider
        if p2[0] > params.room_size[0]:
            # s1 is above s2
            if (r1 == r2).all():
                return self.check_vertical_intersect(p1, p2, params.room_size[0])
            # s1 is below s2
            else:
                return self.check_vertical_intersect((s1[0], p1[1]), (s2[0], p2[1]),
                                                     (r2[0]+1) * params.partition_size[0])
        else:
            # s1 is left of s2
            if (r1 == r2).all():
                return self.check_horizontal_intersect(p1, p2, params.room_size[1])
            # s1 is right of s2
            else:
                return self.check_horizontal_intersect((p1[0], s1[1]), (p2[0], s2[1]),
                                                       (r2[1]+1) * params.partition_size[1])

    # check if the state s is a legal state that is within the grid and not inside any wall area
    # r is the room of the state
    # p is the relative position within the room
    def is_state_legal(self, s, r, p):
        params = self.grid_params

        # make sure state is within the grid
        if not params.grid_region.contains(s):
            return False
        if r[0] >= params.size[0] or r[1] >= params.size[1]:
            return False

        # make sure state is not inside any wall area
        if (p[0] <= params.room_size[0] and p[1] <= params.room_size[1]):
            return True
        elif (p[0] > params.room_size[0] and p[1] >= params.hdoor[0]
              and p[1] <= params.hdoor[1]):
            return params.graph[params.get_index(r)][2]
        elif (p[1] > params.room_size[1] and p[0] >= params.vdoor[0]
              and p[0] <= params.vdoor[1]):
            return params.graph[params.get_index(r)][3]
        else:
            return False

    # check if line from s1 to s2 intersects the horizontal axis at a point inside door region
    # horizontal coordinates should be relative positions within rooms
    def check_vertical_intersect(self, s1, s2, x):
        y = ((s2[1] - s1[1]) * (x - s1[0]) / (s2[0] - s1[0])) + s1[1]
        return (self.grid_params.hdoor[0] <= y
                and y <= self.grid_params.hdoor[1])

    # check if line from s1 to s2 intersects the vertical axis at a point inside door region
    # vertical coordinates should be relative positions within rooms
    def check_horizontal_intersect(self, s1, s2, y):
        x = ((s2[0] - s1[0]) * (y - s1[1]) / (s2[1] - s1[1])) + s1[0]
        return (self.grid_params.vdoor[0] <= x
                and x <= self.grid_params.vdoor[1])
