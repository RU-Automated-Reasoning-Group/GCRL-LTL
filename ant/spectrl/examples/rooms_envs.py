from spectrl.envs.rooms import GridParams

GRID_PARAMS_LIST = []
MAX_TIMESTEPS = []
START_ROOM = []
FINAL_ROOM = []

# parameters for a 3-by-3 grid
size1 = (3, 3)
edges1 = [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((0, 1), (0, 2)),
          ((1, 0), (1, 1)), ((1, 0), (2, 0)), ((1, 1), (1, 2)),
          ((2, 0), (2, 1)), ((2, 1), (2, 2))]
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
