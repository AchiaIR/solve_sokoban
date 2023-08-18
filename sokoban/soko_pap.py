
import gym
from gym.utils import seeding
from gym.spaces.discrete import Discrete
from gym.spaces import Box
import numpy as np
import random
import marshal
import pkg_resources
import imageio

FIXED_IDC = 1

def room_to_rgb(room, room_structure=None):
    """
    Creates an RGB image of the room.
    :param room:
    :param room_structure:
    :return:
    """
    resource_package = __name__

    room = np.array(room)
    if not room_structure is None:
        # Change the ID of a player on a target
        room[(room == 5) & (room_structure == 2)] = 6

    # Load images, representing the corresponding situation
    box_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'box.png')))
    box = imageio.imread(box_filename)

    box_on_target_filename = pkg_resources.resource_filename(resource_package,
                                                             '/'.join(('surface', 'box_on_target.png')))
    box_on_target = imageio.imread(box_on_target_filename)

    box_target_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'box_target.png')))
    box_target = imageio.imread(box_target_filename)

    floor_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'floor.png')))
    floor = imageio.imread(floor_filename)

    player_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'player.png')))
    player = imageio.imread(player_filename)

    player_on_target_filename = pkg_resources.resource_filename(resource_package,
                                                                '/'.join(('surface', 'player_on_target.png')))
    player_on_target = imageio.imread(player_on_target_filename)

    wall_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'wall.png')))
    wall = imageio.imread(wall_filename)

    surfaces = [wall, floor, box_target, box_on_target, box, player, player_on_target]

    # Assemble the new rgb_room, with all loaded images
    room_rgb = np.zeros(shape=(room.shape[0] * 16, room.shape[1] * 16, 3), dtype=np.uint8)
    for i in range(room.shape[0]):
        x_i = i * 16

        for j in range(room.shape[1]):
            y_j = j * 16
            surfaces_id = room[i, j]

            room_rgb[x_i:(x_i + 16), y_j:(y_j + 16), :] = surfaces[surfaces_id]

    return room_rgb


def room_to_tiny_world_rgb(room, room_structure=None, scale=1):

    room = np.array(room)
    if not room_structure is None:
        # Change the ID of a player on a target
        room[(room == 5) & (room_structure == 2)] = 6

    wall = [0, 0, 0]
    floor = [243, 248, 238]
    box_target = [254, 126, 125]
    box_on_target = [254, 95, 56]
    box = [142, 121, 56]
    player = [160, 212, 56]
    player_on_target = [219, 212, 56]

    surfaces = [wall, floor, box_target, box_on_target, box, player, player_on_target]

    # Assemble the new rgb_room, with all loaded images
    room_small_rgb = np.zeros(shape=(room.shape[0]*scale, room.shape[1]*scale, 3), dtype=np.uint8)
    for i in range(room.shape[0]):
        x_i = i * scale
        for j in range(room.shape[1]):
            y_j = j * scale
            surfaces_id = int(room[i, j])
            room_small_rgb[x_i:(x_i+scale), y_j:(y_j+scale), :] = np.array(surfaces[surfaces_id])

    return room_small_rgb


def room_to_rgb_FT(room, box_mapping, room_structure=None):
    """
    Creates an RGB image of the room.
    :param room:
    :param room_structure:
    :return:
    """
    resource_package = __name__

    room = np.array(room)
    if not room_structure is None:
        # Change the ID of a player on a target
        room[(room == 5) & (room_structure == 2)] = 6

    # Load images, representing the corresponding situation
    box_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'box.png')))
    box = imageio.imread(box_filename)

    box_on_target_filename = pkg_resources.resource_filename(resource_package,
                                                             '/'.join(('surface', 'box_on_target.png')))
    box_on_target = imageio.imread(box_on_target_filename)

    box_target_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'box_target.png')))
    box_target = imageio.imread(box_target_filename)

    floor_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'floor.png')))
    floor = imageio.imread(floor_filename)

    player_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'player.png')))
    player = imageio.imread(player_filename)

    player_on_target_filename = pkg_resources.resource_filename(resource_package,
                                                                '/'.join(('surface', 'player_on_target.png')))
    player_on_target = imageio.imread(player_on_target_filename)

    wall_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'wall.png')))
    wall = imageio.imread(wall_filename)

    surfaces = [wall, floor, box_target, box_on_target, box, player, player_on_target]

    # Assemble the new rgb_room, with all loaded images
    room_rgb = np.zeros(shape=(room.shape[0] * 16, room.shape[1] * 16, 3), dtype=np.uint8)
    for i in range(room.shape[0]):
        x_i = i * 16

        for j in range(room.shape[1]):
            y_j = j * 16

            surfaces_id = room[i, j]
            surface = surfaces[surfaces_id]
            if 1 < surfaces_id < 5:
                try:
                    surface = get_proper_box_surface(surfaces_id, box_mapping, i, j)
                except:
                    pass
            room_rgb[x_i:(x_i + 16), y_j:(y_j + 16), :] = surface

    return room_rgb


def get_proper_box_surface(surfaces_id, box_mapping, i, j):
    # not used, kept for documentation
    # names = ["wall", "floor", "box_target", "box_on_target", "box", "player", "player_on_target"]
    
    box_id = 0
    situation = ''

    if surfaces_id == 2:
        situation = '_target'
        box_id = list(box_mapping.keys()).index((i, j))
    elif surfaces_id == 3:
        box_id = list(box_mapping.values()).index((i, j))
        box_key = list(box_mapping.keys())[box_id]
        if box_key == (i, j):
            situation = '_on_target'
        else:
            situation = '_on_wrong_target'
        pass
    elif surfaces_id == 4:
        box_id = list(box_mapping.values()).index((i, j))

    surface_name = 'box{}{}.png'.format(box_id, situation)
    resource_package = __name__
    filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'multibox', surface_name)))
    surface = imageio.imread(filename)

    return surface


def room_to_tiny_world_rgb_FT(room, box_mapping, room_structure=None, scale=1):
        room = np.array(room)
        if not room_structure is None:
            # Change the ID of a player on a target
            room[(room == 5) & (room_structure == 2)] = 6

        wall = [0, 0, 0]
        floor = [243, 248, 238]
        box_target = [254, 126, 125]
        box_on_target = [254, 95, 56]
        box = [142, 121, 56]
        player = [160, 212, 56]
        player_on_target = [219, 212, 56]

        surfaces = [wall, floor, box_target, box_on_target, box, player, player_on_target]

        # Assemble the new rgb_room, with all loaded images
        room_small_rgb = np.zeros(shape=(room.shape[0] * scale, room.shape[1] * scale, 3), dtype=np.uint8)
        for i in range(room.shape[0]):
            x_i = i * scale
            for j in range(room.shape[1]):
                y_j = j * scale

                surfaces_id = int(room[i, j])
                surface = np.array(surfaces[surfaces_id])
                if 1 < surfaces_id < 5:
                    try:
                        surface = get_proper_tiny_box_surface(surfaces_id, box_mapping, i, j)
                    except:
                        pass
                room_small_rgb[x_i:(x_i + scale), y_j:(y_j + scale), :] = surface

        return room_small_rgb


def get_proper_tiny_box_surface(surfaces_id, box_mapping, i, j):

    box_id = 0
    situation = 'box'

    if surfaces_id == 2:
        situation = 'target'
        box_id = list(box_mapping.keys()).index((i, j))
    elif surfaces_id == 3:
        box_id = list(box_mapping.values()).index((i, j))
        box_key = list(box_mapping.keys())[box_id]
        if box_key == (i, j):
            situation = 'on_target'
        else:
            situation = 'on_wrong_target'
        pass
    elif surfaces_id == 4:
        box_id = list(box_mapping.values()).index((i, j))

    surface = [255, 255, 255]
    if box_id == 0:
        if situation == 'target':
            surface = [111, 127, 232]
        elif situation == 'on_target':
            surface = [6, 33, 130]
        elif situation == 'on_wrong_target':
            surface = [69, 81, 122]
        else:
            # Just the box
            surface = [11, 60, 237]

    elif box_id == 1:
        if situation == 'target':
            surface = [195, 127, 232]
        elif situation == 'on_target':
            surface = [96, 5, 145]
        elif situation == 'on_wrong_target':
            surface = [96, 63, 114]
        else:
            surface = [145, 17, 214]

    elif box_id == 2:
        if situation == 'target':
            surface = [221, 113, 167]
        elif situation == 'on_target':
            surface = [140, 5, 72]
        elif situation == 'on_wrong_target':
            surface = [109, 60, 71]
        else:
            surface = [239, 0, 55]

    elif box_id == 3:
        if situation == 'target':
            surface = [247, 193, 145]
        elif situation == 'on_target':
            surface = [132, 64, 3]
        elif situation == 'on_wrong_target':
            surface = [94, 68, 46]
        else:
            surface = [239, 111, 0]

    return surface


def color_player_two(room_rgb, position, room_structure):
    resource_package = __name__

    player_filename = pkg_resources.resource_filename(resource_package, '/'.join(('surface', 'multiplayer', 'player1.png')))
    player = imageio.imread(player_filename)

    player_on_target_filename = pkg_resources.resource_filename(resource_package,
                                                                '/'.join(('surface', 'multiplayer', 'player1_on_target.png')))
    player_on_target = imageio.imread(player_on_target_filename)

    x_i = position[0] * 16
    y_j = position[1] * 16

    if room_structure[position[0], position[1]] == 2:
        room_rgb[x_i:(x_i + 16), y_j:(y_j + 16), :] = player_on_target

    else:
        room_rgb[x_i:(x_i + 16), y_j:(y_j + 16), :] = player

    return room_rgb


def color_tiny_player_two(room_rgb, position, room_structure, scale = 4):

    x_i = position[0] * scale
    y_j = position[1] * scale

    if room_structure[position[0], position[1]] == 2:
        room_rgb[x_i:(x_i + scale), y_j:(y_j + scale), :] = [195, 127, 232]

    else:
        room_rgb[x_i:(x_i + scale), y_j:(y_j + scale), :] = [96, 5, 145]

    return room_rgb


TYPE_LOOKUP = {
    0: 'wall',
    1: 'empty space',
    2: 'box target',
    3: 'box on target',
    4: 'box not on target',
    5: 'player'
}

ACTION_LOOKUP = {
    0: 'push up',
    1: 'push down',
    2: 'push left',
    3: 'push right',
    4: 'move up',
    5: 'move down',
    6: 'move left',
    7: 'move right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}
def generate_room(dim=(13, 13), p_change_directions=0.35, num_steps=25, num_boxes=3, tries=4, second_player=False):
    """
    Generates a Sokoban room, represented by an integer matrix. The elements are encoded as follows:
    wall = 0
    empty space = 1
    box target = 2
    box not on target = 3
    box on target = 4
    player = 5
    :param dim:
    :param p_change_directions:
    :param num_steps:
    :return: Numpy 2d Array
    """
    room_state = np.zeros(shape=dim)
    room_structure = np.zeros(shape=dim)

    # Some times rooms with a score == 0 are the only possibility.
    # In these case, we try another model.
    for t in range(tries):
        room = room_topology_generation(dim, p_change_directions, num_steps)
        room = place_boxes_and_player(room, num_boxes=num_boxes, second_player=second_player)

        # Room fixed represents all not movable parts of the room
        room_structure = np.copy(room)
        room_structure[room_structure == 5] = 1

        # Room structure represents the current state of the room including movable parts
        room_state = room.copy()
        room_state[room_state == 2] = 4

        room_state, score, box_mapping = reverse_playing(room_state, room_structure)
        room_state[room_state == 3] = 4

        if score > 0:
            break

    if score == 0:
        raise RuntimeWarning('Generated Model with score == 0')

    return room_structure, room_state, box_mapping


def room_topology_generation(dim=(10, 10), p_change_directions=0.35, num_steps=15):
    """
    Generate a room topology, which consits of empty floors and walls.
    :param dim:
    :param p_change_directions:
    :param num_steps:
    :return:
    """
    dim_x, dim_y = dim

    # The ones in the mask represent all fields which will be set to floors
    # during the random walk. The centered one will be placed over the current
    # position of the walk.
    masks = [
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ],
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ],
        [
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 0]
        ],
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 0]
        ]
    ]
    
    np.random.seed(0)

    # Possible directions during the walk
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    direction = random.sample(directions, 1)[0]

    np.random.seed(0)

    # Starting position of random walk
    position = np.array([
        random.randint(1, dim_x - 1),
        random.randint(1, dim_y - 1)]
    )
    ####################################
    #if FIXED_IDC:
    #  print("FIXED_IDC")
    #  print("direction = ", direction)  
    #  print("position=", position)

    ####################################
    level = np.zeros(dim, dtype=int)

    for s in range(num_steps):

        # Change direction randomly
        if random.random() < p_change_directions:
            direction = random.sample(directions, 1)[0]

        # Update position
        position = position + direction
        position[0] = max(min(position[0], dim_x - 2), 1)
        position[1] = max(min(position[1], dim_y - 2), 1)

        # Apply mask
        mask = random.sample(masks, 1)[0]
        mask_start = position - 1
        level[mask_start[0]:mask_start[0] + 3, mask_start[1]:mask_start[1] + 3] += mask

    level[level > 0] = 1
    level[:, [0, dim_y - 1]] = 0
    level[[0, dim_x - 1], :] = 0

    return level


def place_boxes_and_player(room, num_boxes, second_player):
    """
    Places the player and the boxes into the floors in a room.
    :param room:
    :param num_boxes:
    :return:
    """
    # Get all available positions
    possible_positions = np.where(room == 1)
    num_possible_positions = possible_positions[0].shape[0]
    num_players = 2 if second_player else 1

    if num_possible_positions <= num_boxes + num_players:
        raise RuntimeError('Not enough free spots (#{}) to place {} player and {} boxes.'.format(
            num_possible_positions,
            num_players,
            num_boxes)
        )

    # Place player(s)
    ind = np.random.randint(num_possible_positions)
    player_position = possible_positions[0][ind], possible_positions[1][ind]
    #if FIXED_IDC:
    #  print(player_position)
    room[player_position] = 5

    if second_player:
        ind = np.random.randint(num_possible_positions)
        player_position = possible_positions[0][ind], possible_positions[1][ind]
        room[player_position] = 5

    # Place boxes
    for n in range(num_boxes):
        possible_positions = np.where(room == 1)
        num_possible_positions = possible_positions[0].shape[0]

        ind = np.random.randint(num_possible_positions)
        box_position = possible_positions[0][ind], possible_positions[1][ind]
        #if FIXED_IDC:
        #  print(box_position)
        room[box_position] = 2

    return room


# Global variables used for reverse playing.
explored_states = set()
num_boxes = 0
best_room_score = -1
best_room = None
best_box_mapping = None


def reverse_playing(room_state, room_structure, search_depth=100):
    """
    This function plays Sokoban reverse in a way, such that the player can
    move and pull boxes.
    It ensures a solvable level with all boxes not being placed on a box target.
    :param room_state:
    :param room_structure:
    :param search_depth:
    :return: 2d array
    """
    global explored_states, num_boxes, best_room_score, best_room, best_box_mapping

    # Box_Mapping is used to calculate the box displacement for every box
    box_mapping = {}
    box_locations = np.where(room_structure == 2)
    num_boxes = len(box_locations[0])
    for l in range(num_boxes):
        box = (box_locations[0][l], box_locations[1][l])
        box_mapping[box] = box

    # explored_states globally stores the best room state and score found during search
    explored_states = set()
    best_room_score = -1
    best_box_mapping = box_mapping
    depth_first_search(room_state, room_structure, box_mapping, box_swaps=0, last_pull=(-1, -1), ttl=300)

    return best_room, best_room_score, best_box_mapping


def depth_first_search(room_state, room_structure, box_mapping, box_swaps=0, last_pull=(-1, -1), ttl=300):
    """
    Searches through all possible states of the room.
    This is a recursive function, which stops if the tll is reduced to 0 or
    over 1.000.000 states have been explored.
    :param room_state:
    :param room_structure:
    :param box_mapping:
    :param box_swaps:
    :param last_pull:
    :param ttl:
    :return:
    """
    global explored_states, num_boxes, best_room_score, best_room, best_box_mapping

    ttl -= 1
    if ttl <= 0 or len(explored_states) >= 300000:
        return

    state_tohash = marshal.dumps(room_state)

    # Only search this state, if it not yet has been explored
    if not (state_tohash in explored_states):

        # Add current state and its score to explored states
        room_score = box_swaps * box_displacement_score(box_mapping)
        if np.where(room_state == 2)[0].shape[0] != num_boxes:
            room_score = 0

        if room_score > best_room_score:
            best_room = room_state
            best_room_score = room_score
            best_box_mapping = box_mapping

        explored_states.add(state_tohash)

        for action in ACTION_LOOKUP.keys():
            # The state and box mapping  need to be copied to ensure
            # every action start from a similar state.
            room_state_next = room_state.copy()
            box_mapping_next = box_mapping.copy()

            room_state_next, box_mapping_next, last_pull_next = \
                reverse_move(room_state_next, room_structure, box_mapping_next, last_pull, action)

            box_swaps_next = box_swaps
            if last_pull_next != last_pull:
                box_swaps_next += 1

            depth_first_search(room_state_next, room_structure,
                               box_mapping_next, box_swaps_next,
                               last_pull, ttl)


def reverse_move(room_state, room_structure, box_mapping, last_pull, action):
    """
    Perform reverse action. Where all actions in the range [0, 3] correspond to
    push actions and the ones greater 3 are simmple move actions.
    :param room_state:
    :param room_structure:
    :param box_mapping:
    :param last_pull:
    :param action:
    :return:
    """
    player_position = np.where(room_state == 5)
    player_position = np.array([player_position[0][0], player_position[1][0]])

    change = CHANGE_COORDINATES[action % 4]
    next_position = player_position + change

    # Check if next position is an empty floor or an empty box target
    if room_state[next_position[0], next_position[1]] in [1, 2]:

        # Move player, independent of pull or move action.
        room_state[player_position[0], player_position[1]] = room_structure[player_position[0], player_position[1]]
        room_state[next_position[0], next_position[1]] = 5

        # In addition try to pull a box if the action is a pull action
        if action < 4:
            possible_box_location = change[0] * -1, change[1] * -1
            possible_box_location += player_position

            if room_state[possible_box_location[0], possible_box_location[1]] in [3, 4]:
                # Perform pull of the adjacent box
                room_state[player_position[0], player_position[1]] = 3
                room_state[possible_box_location[0], possible_box_location[1]] = room_structure[
                    possible_box_location[0], possible_box_location[1]]

                # Update the box mapping
                for k in box_mapping.keys():
                    if box_mapping[k] == (possible_box_location[0], possible_box_location[1]):
                        box_mapping[k] = (player_position[0], player_position[1])
                        last_pull = k

    return room_state, box_mapping, last_pull


def box_displacement_score(box_mapping):
    """
    Calculates the sum of all Manhattan distances, between the boxes
    and their origin box targets.
    :param box_mapping:
    :return:
    """
    score = 0
    
    for box_target in box_mapping.keys():
        box_location = np.array(box_mapping[box_target])
        box_target = np.array(box_target)
        dist = np.sum(np.abs(box_location - box_target))
        score += dist

    return score


TYPE_LOOKUP = {
    0: 'wall',
    1: 'empty space',
    2: 'box target',
    3: 'box on target',
    4: 'box not on target',
    5: 'player'
}

ACTION_LOOKUP = {
    0: 'push up',
    1: 'push down',
    2: 'push left',
    3: 'push right',
    4: 'move up',
    5: 'move down',
    6: 'move left',
    7: 'move right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}



class SokobanEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw']
    }

    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=120,
                 num_boxes=4,
                 num_gen_steps=None,
                 reset=True):

        # General Configuration
        self.dim_room = dim_room
        if num_gen_steps == None:
            self.num_gen_steps = int(1.7 * (dim_room[0] + dim_room[1]))
        else:
            self.num_gen_steps = num_gen_steps

        self.num_boxes = num_boxes
        self.boxes_on_target = 0

        # Penalties and Rewards
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -1
        self.reward_box_on_target = 1
        self.reward_finished = 10
        self.reward_last = 0

        # Other Settings
        self.viewer = None
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        
        if reset:
            # Initialize Room
            _ = self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        #return [seed]
        return 70

    def step(self, action, observation_mode='rgb_array'):
        assert action in ACTION_LOOKUP
        assert observation_mode in ['rgb_array', 'tiny_rgb_array', 'raw']

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False

        if action == 0:
            moved_player = False

        # All push actions are in the range of [0, 3]
        elif action < 5:
            moved_player, moved_box = self._push(action)

        else:
            moved_player = self._move(action)

        self._calc_reward()
        
        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.render(mode=observation_mode)

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()

        return observation, self.reward_last, done, info

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] \
                or new_box_position[1] >= self.room_state.shape[1]:
            return False, False


        can_push_box = self.room_state[new_position[0], new_position[1]] in [3, 4]
        can_push_box &= self.room_state[new_box_position[0], new_box_position[1]] in [1, 2]
        if can_push_box:

            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            # Move Box
            box_type = 4
            if self.room_fixed[new_box_position[0], new_box_position[1]] == 2:
                box_type = 3
            self.room_state[new_box_position[0], new_box_position[1]] = box_type
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            return True

        return False

    def _calc_reward(self):
        """
        Calculate Reward Based on
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        # count boxes off or on the target
        empty_targets = self.room_state == 2
        player_on_target = (self.room_fixed == 2) & (self.room_state == 5)
        total_targets = empty_targets | player_on_target

        current_boxes_on_target = self.num_boxes - \
                                  np.where(total_targets)[0].shape[0]

        # Add the reward if a box is pushed on the target and give a
        # penalty if a box is pushed off the target.
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.boxes_on_target:
            self.reward_last += self.penalty_box_off_target
        
        game_won = self._check_if_all_boxes_on_target()        
        if game_won:
            self.reward_last += self.reward_finished
        
        self.boxes_on_target = current_boxes_on_target

    def _check_if_done(self):
        # Check if the game is over either through reaching the maximum number
        # of available steps or by pushing all boxes on the targets.        
        return self._check_if_all_boxes_on_target() or self._check_if_maxsteps()

    def _check_if_all_boxes_on_target(self):
        empty_targets = self.room_state == 2
        player_hiding_target = (self.room_fixed == 2) & (self.room_state == 5)
        are_all_boxes_on_targets = np.where(empty_targets | player_hiding_target)[0].shape[0] == 0
        return are_all_boxes_on_targets

    def _check_if_maxsteps(self):
        return (self.max_steps == self.num_env_steps)

    def reset(self, second_player=False, render_mode='rgb_array'):
        try:
            self.room_fixed, self.room_state, self.box_mapping = generate_room(
                dim=self.dim_room,
                num_steps=self.num_gen_steps,
                num_boxes=self.num_boxes,
                second_player=second_player
            )
        except (RuntimeError, RuntimeWarning) as e:
            print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
            print("[SOKOBAN] Retry . . .")
            return self.reset(second_player=second_player, render_mode=render_mode)

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = self.render(render_mode)
        return starting_observation

    def render(self, mode='human', close=None, scale=1):
        assert mode in RENDERING_MODES

        img = self.get_image(mode, scale)

        if 'rgb_array' in mode:
            return img

        elif 'human' in mode:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

        elif 'raw' in mode:
            arr_walls = (self.room_fixed == 0).view(np.int8)
            arr_goals = (self.room_fixed == 2).view(np.int8)
            arr_boxes = ((self.room_state == 4) + (self.room_state == 3)).view(np.int8)
            arr_player = (self.room_state == 5).view(np.int8)

            return arr_walls, arr_goals, arr_boxes, arr_player

        else:
            super(SokobanEnv, self).render(mode=mode)  # just raise an exception

    def get_image(self, mode, scale=1):
        
        if mode.startswith('tiny_'):
            img = room_to_tiny_world_rgb(self.room_state, self.room_fixed, scale=scale)
        else:
            img = room_to_rgb(self.room_state, self.room_fixed)

        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def set_maxsteps(self, num_steps):
        self.max_steps = num_steps

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP


ACTION_LOOKUP = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right',
    5: 'move up',
    6: 'move down',
    7: 'move left',
    8: 'move right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'raw']




class PushAndPullSokobanEnv(SokobanEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw']
    }

    def __init__(self,
             dim_room=(10, 10),
             max_steps=120,
             num_boxes=3,
             num_gen_steps=None):

        super(PushAndPullSokobanEnv, self).__init__(dim_room, max_steps, num_boxes, num_gen_steps)
        screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3))
        self.boxes_are_on_target = [False] * num_boxes
        self.action_space = Discrete(len(ACTION_LOOKUP))
        
        _ = self.reset()

    def step(self, action, observation_mode='rgb_array'):
        assert action in ACTION_LOOKUP

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False
        if action == 0:
            moved_player = False

        # All push actions are in the range of [0, 3]
        if action < 5:
            moved_player, moved_box = self._push(action)

        elif action < 9:
            moved_player = self._move(action)

        else:
            moved_player, moved_box = self._pull(action)

        self._calc_reward()

        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.render(mode=observation_mode)

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()

        return observation, self.reward_last, done, info

    def _pull(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()
        pull_content_position = self.player_position - change

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            box_next_to_player = self.room_state[pull_content_position[0], pull_content_position[1]] in [3, 4]
            if box_next_to_player:
                # Move Box
                box_type = 4
                if self.room_fixed[current_position[0], current_position[1]] == 2:
                    box_type = 3
                self.room_state[current_position[0], current_position[1]] = box_type
                self.room_state[pull_content_position[0], pull_content_position[1]] = \
                    self.room_fixed[pull_content_position[0], pull_content_position[1]]

            return True, box_next_to_player

        return False, False

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP


ACTION_LOOKUP = {
    0: 'no operation',
    1: 'push up',
    2: 'push down',
    3: 'push left',
    4: 'push right',
    5: 'move up',
    6: 'move down',
    7: 'move left',
    8: 'move right',
    9: 'pull up',
    10: 'pull down',
    11: 'pull left',
    12: 'pull right',
}