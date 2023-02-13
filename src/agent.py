                                    #####################################
                                    #@$%&                           &%$@#
                                    #!     AGS - Projekt TileWorld     !#
                                    #!           Michal Glos           !#
                                    #!            xglosm01             !#
                                    #!              __                 !#
                                    #!            <(o )___             !#
                                    #!             ( ._> /             !#
                                    #!              `---'              !#
                                    #@$%&                           &%$@#
                                    #####################################

import numpy as np
import utils
import time
import pprint
from functools import lru_cache

MAX_STANBY_TIME = 3
AGENT = 3
AGENT_STANDBY = 4
COULD_SEE_CACHE = {}
DISTANCE_CACHE = {}
times = {}

def measure(fn):
    def timed(*args, **kwargs):
        start = time.time()
        res = fn(*args, **kwargs)
        fn_time = time.time() - start
        key = fn.__name__
        if key in times:
            times[key] += fn_time
        else:
            times[key] = fn_time
        return res
    return timed

class Agent:
    '''
    Agent entity discovering tiles on board
    Provides several step types, based on strategy

    Important properties:
    self.moves:     Allowed moves
    self.pos:       Actual position of agent
    self.dims:      Dimensions of searched board
    self.step:      Function for iteration of agent search
    '''
    def __init__(self, agent_type, pos, dims, move_diagonally, agent_vision):
        '''Initialize agent class'''
        # restrict diagonal moves if requested
        if move_diagonally:
            self.moves = np.array([[x,y] for x in [-1,0,1] for y in [-1,0,1] if any([x,y])])
        else:
            self.moves = np.array([[x,y] for x in [-1,0,1] for y in [-1,0,1] if any([x,y]) and not all([x,y])])
        self.pos = pos
        self.standby_time = 0
        self.agent_vision = agent_vision
        self.dims = dims    # Board dimensions
        self._color = AGENT_STANDBY # At first, agent does not move
        self.move_diagonally = move_diagonally
        # Get himself a nice random number generator
        self.rng = np.random.default_rng()
        if agent_type == 'random':
            # Initialize random movement agent
            self.step = self.random
        elif agent_type == 'naive':
            # Inititalize naive agent searching for closest not discovered tile
            # Initialize with empty target (is tested for len, if target active or needs to be determined)
            self.old_pos = np.array([pos])
            self.target = []
            self.get_random = False
            self.step = self.naive
        elif agent_type == 'smart':
            self.path = None
            self.target = None
            self.step = self.smart
        elif agent_type == 'smart_coop':
            self.depth = int(self.agent_vision)
            self.path = None
            self.target = pos
            self.step = self.smart_coop

    ######################### Properties #########################

    @property
    def position(self):
        return self.pos

    @property
    def color(self):
        return self._color

    ######################### Utilities #########################

    @measure
    def finished(self):
        '''If agent finished, change it's color to standby'''
        self._color = AGENT_STANDBY
        self.standby_time += 1

    @measure
    def concat_restrictions(self, obstacles, agents):
        '''Compose restricted tiles from agents and obstacles positions'''
        p_agents = np.array([a.position for a in agents])
        if len(obstacles) == 0:
            return p_agents
        else:
            return np.append(obstacles, p_agents, axis=0)

    @measure
    def expand_positions(self, omit, position=None):
        '''Get possible moves from position. Do not expand into omit positions'''
        if position is None:
            position = self.pos
        positions = np.array([position + move for move in self.moves])
        path = np.array([pos for pos in positions if utils.in_board(pos, self.dims) and not utils.is_element_in_array(omit, pos)])
        return path

    @measure
    def get_distances(self, pos=None):
        '''Get distance to each tile on board from position'''
        if pos is None:
            pos = self.pos
        if pos[0] not in DISTANCE_CACHE:
            DISTANCE_CACHE[pos[0]] = {}
        if pos[1] not in DISTANCE_CACHE[pos[0]]:
            indexes = np.array(np.where(np.zeros(self.dims) == 0))
            dx = np.absolute(indexes[0] - pos[0])
            dy = np.absolute(indexes[1] - pos[1])
            # Diagonal distances
            if self.move_diagonally:
                distances = np.maximum(dx, dy)
            # Manhatton distances
            else:
                distances = dx + dy
            DISTANCE_CACHE[pos[0]][pos[1]] = distances.reshape(self.dims)
        return DISTANCE_CACHE[pos[0]][pos[1]]

    @measure
    def get_not_discovered_distances(self, discovered):
        '''Get distances from each not discovered tile. Returns array of 3 element vector (x,y,distance)'''
        not_discovered_distances = self.get_distances() * (discovered - 1)
        points = np.array(np.where(not_discovered_distances != 0)).transpose()
        not_discovered_distances = not_discovered_distances[not_discovered_distances != 0] * -1
        ids_sort = np.argsort(not_discovered_distances)
        return np.column_stack((points, not_discovered_distances))[ids_sort].astype(int)

    @measure
    def get_closest_not_discovered(self, discovered):
        '''Get array of closest not discovered tiles. take into account diagonal'''
        distances = self.get_distances() * (discovered - 1)
        min_dist = (distances[np.nonzero(distances)]).max()
        return self.rng.choice(np.array(np.where(distances == min_dist)).transpose(), 1)[0]
        
    @measure
    def naive_move_to_target(self, restrictions):
        '''Move naively, do not plan overcoming obstacles, but still, do not step on them'''
        # Find possible positions
        possible_positions = self.expand_positions(restrictions)
        # If none found (can't move), do not move ...
        if len(possible_positions) == 0:
            return self.pos
        # Find distance from target for each position, sort it from lowest distance
        cost = np.array([abs(self.target[0]-pos[0])+abs(self.target[1]-pos[1]) for pos in possible_positions])
        sort_args = np.argsort(cost)
        # Return the closest one
        return possible_positions[sort_args[0]]

    @measure
    def could_see_matrix(self, pos):
        key = f'{pos[0]},{pos[1]}'
        if key not in COULD_SEE_CACHE:
            min_x, max_x = max(0, pos[0]-int(self.agent_vision)), min(self.dims[0]-1, pos[0] + int(self.agent_vision))
            min_y, max_y = max(0, pos[1]-int(self.agent_vision)), min(self.dims[1]-1, pos[1] + int(self.agent_vision))
            could_see = np.array([utils.could_see(x-pos[0], y-pos[1], self.agent_vision) for x in range(min_x, max_x + 1) \
                                       for y in range(min_y, max_y + 1)]).reshape((max_x-min_x+1, max_y-min_y+1))
            COULD_SEE_CACHE[key] = ((min_x, max_x+1, min_y, max_y+1), could_see)
        return COULD_SEE_CACHE[key]


    @measure
    def bfs_closest(self, restrictions, discovered, border=False):
        '''Apply BFS to find closest reachable tile'''
        if len(restrictions) == 0:
            restrictions = np.array([[-2,-2]])
        to_expand = [self.pos]
        expanded = [[-1,-1]]
        next_to_expand = []
        undiscovered = []
        not_discovered = []
        depth = 0
        while (not len(not_discovered) and not border) or (border and len(to_expand) > 0):
            omit = np.append(to_expand, expanded, axis=0)
            omit = np.append(omit, restrictions, axis=0)
            next_to_expand = np.array([_pos for pos in to_expand for _pos in self.expand_positions(omit, position=pos) if discovered[_pos[0], _pos[1]]]).reshape((-1,2))
            next_to_expand = np.unique(next_to_expand.view(np.dtype(
                    (np.void, next_to_expand.dtype.itemsize*next_to_expand.shape[1])
                ))).view(next_to_expand.dtype).reshape(-1, next_to_expand.shape[1])
            do_not_expand = []
            for pos in next_to_expand:
                # Construct agent vision field
                (x, _x, y, _y), could_see_matrix = self.could_see_matrix(pos)
                # If agent can see undiscovered tile
                undiscovered = discovered[x:_x, y:_y][could_see_matrix]
                if len(undiscovered) > 0:
                    gain = np.count_nonzero(undiscovered == 0)
                    if gain > 0:
                        not_discovered.append((pos, gain, depth))
                        do_not_expand.append(True)
                    do_not_expand.append(False)
            if len(undiscovered) > 0:
                next_to_expand = [pos for i, pos in enumerate(next_to_expand) if not do_not_expand[i]]
            if len(next_to_expand) == 0:
                return not_discovered if len(not_discovered) else None
            expanded = omit
            if border:
                to_expand = [node for node in next_to_expand if discovered[node[0], node[1]]]
            else:
                to_expand = next_to_expand
            depth += 1
        return not_discovered

    def get_distance(self, pos1, pos2):
        '''Get distance from 2 positions'''
        if self.move_diagonally:
            return np.absolute(pos1-pos2).max()
        else:
            return np.absolute(pos1-pos2).sum()

    @measure
    def a_star(self, restrictions):
        '''Create array of positions to reach target
        Given a target, use a* algorithm to compute the shortest path'''
        # Position, distance, pointer to previous position,expanded?
        positions = np.array([(self.pos, self.get_distance(self.pos, self.target), 0, False)], dtype='object')
        pos = positions[0][0]
        # If restrictions empty, do not break the code
        if not restrictions.shape[0]:
            restrictions = np.array([[-2,-2]])
        # Not empty in order to hold shape
        expanded = np.array([[-1,-1]])
        # Expand first node
        pos_id = 0
        while (pos != self.target).any():
            expanded = np.array([pos[0] for pos in positions])
            positions = np.append(positions, np.array([(pos, self.get_distance(pos, self.target), pos_id, False) \
                                              for pos in self.expand_positions(np.append(restrictions, expanded, axis=0), position=pos)], dtype='object')).reshape((-1,4))
            positions[pos_id][3] = True
            if all([p[3] for p in positions]):
                return False
            # Next predecessor ID (shortest distance)
            pos_id = np.apply_along_axis(lambda x: x[1] if not x[3] else np.prod(self.dims), axis=1, arr=positions).argsort()[0]
            pos = positions[pos_id][0]
        # Compose the path
        position = positions[pos_id]
        self.path = []
        self.path.append(position[0])
        while position[2]:
            position = positions[position[2]]
            self.path.append(position[0])
        self.path = np.array(self.path)


    @measure
    def path_step(self, discovered, restrictions, o, a):
        '''Take one step from path of positions'''
        pos = self.path[-1]
        if not utils.is_element_in_array(restrictions, pos):
            self.path = np.delete(self.path, -1, 0)
            self.pos = pos
        else:
            # If stuck (because of another agent, have 50% chance to move somewhere else)
            # this works a little
            if np.random.random() > 0.5:
                pos = self.expand_positions(restrictions)
                if len(pos):
                    self.path = np.append(self.path, [self.pos], axis=0)
                    self.pos = pos[0]
            else:
                # DO not count as classicaly stuck agent, just obstructed with another agent
                if utils.is_element_in_array(o, pos):
                    self.path = None
                    print('wtf')
                    self.step(o, a, discovered)
                else:
                    pass

    @measure
    def agent_trapped(self, restrictions, discovered):
        '''With 25% determine if agent got trapped'''
        if self.standby_time or np.random.random() < 0.25:
            closest_reachable_tile = self.bfs_closest(restrictions, discovered)
            if closest_reachable_tile is None:
                self.finished()
            else:
                self.standby_time = 0


    @measure
    def most_distanced_target(self, possible_targets, agents):
        '''Choose target with highest distance from other agents targets and lowest distance from position'''
        positions = [(a.target, a.pos) for a in agents if not (a.pos == self.pos).all()]
        _possible_targets = [t[0] for t in possible_targets]
        distances = [[((t[0]-p_t[0])**2+(t[1]+p_t[1])**2)**(1/2) for t, p in positions] for p_t in _possible_targets]
        values = np.array([min(v) for v in distances])
        # Choose targets far enough, do not interfere in each others vision
        targets =  [possible_targets[i] for i, b in enumerate(values > 2*self.agent_vision) if b]
        if len(targets) == 0:
            targets = possible_targets
        close_targets = sorted(targets, key=lambda x: (x[2], -x[1]))
        if len(close_targets) == 0:
            close_targets = targets
        return close_targets[0][0]


    @measure
    def target_discovered(self, discovered):
        (x, _x, y, _y), could_see_matrix = self.could_see_matrix(self.target)
        # If agent can see undiscovered tile
        return (discovered[x:_x, y:_y][could_see_matrix]).all()

    ######################### different steps #########################

    @measure
    def smart_coop(self, obstacles, agents, discovered):
        # Get tiles restricted from movement
        restrictions = self.concat_restrictions(obstacles, agents)
        self._color = AGENT
        # If path does not exist or target tile was already discovered
        if self.path is None or not len(self.path) or self.target_discovered(discovered):
            closest_reachable_tile = self.bfs_closest(obstacles, discovered, border=True)
            if closest_reachable_tile is None:
                self.finished()
                return self.pos
            else:
                self.standby_time = 0
            # Choose highest gain positions
            self.target = self.most_distanced_target(closest_reachable_tile, agents)
            self.a_star(obstacles)
        self.path_step(discovered, restrictions, obstacles, agents)
        return self.pos

    @measure
    def smart(self, obstacles, agents, discovered):
        # Get tiles restricted from movement
        restrictions = self.concat_restrictions(obstacles, agents)
        self._color = AGENT
        # If path does not exist or target tile was already discovered
        if self.path is None or not len(self.path) or self.target_discovered(discovered):
            closest_reachable_tile = self.bfs_closest(obstacles, discovered)
            if closest_reachable_tile is None:
                self.finished()
                return self.pos
            else:
                self.standby_time = 0
            # Choose highest gain positions
            self.target = sorted(closest_reachable_tile, key=lambda x: x[1])[-1][0]
            self.a_star(obstacles)
        self.path_step(discovered, restrictions, obstacles, agents)
        return self.pos

    @measure
    def random(self, obstacles, agents, discovered):
        # Get tiles restricted from movement
        restrictions = self.concat_restrictions(obstacles, agents)
        self.agent_trapped(restrictions, discovered)
        # calculate possible moves
        possible_positions = self.expand_positions(restrictions)
        # Choose a move
        if len(possible_positions) == 0:
            self.finished()
        else:
            self.pos = self.rng.choice(possible_positions, 1)[0]
            self._color = AGENT

    @measure
    def naive(self, obstacles, agents, discovered):
        '''Cant plan with obstacles'''
        # Get tiles restricted from movement
        restrictions = self.concat_restrictions(obstacles, agents)
        self.agent_trapped(restrictions, discovered)
        if self.get_random:
            self.random(obstacles, agents, discovered)
            # 25% chance to get out of random movement
            if np.random.random() < 0.25:
                self.get_random = False
        else:
            # If not target or target discovered, get one.
            if not len(self.target) or discovered[self.target[0], self.target[1]]:
                # Choose target randomly from closest not discovered tiles
                not_discovered = self.get_closest_not_discovered(discovered)
                # If all discovered, stay in place
                if len(not_discovered) == 0:
                    self.finished()
                    return
                # Choose target randomly
                self.target = not_discovered

            self.pos = self.naive_move_to_target(restrictions)
            self.old_pos = np.append(self.old_pos, [self.pos], axis=0)
            if len(self.old_pos) > 3 and (self.old_pos[-3] == self.pos).all():
                self.get_random = True
            self._color = AGENT
