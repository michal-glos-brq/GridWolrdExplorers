import numpy as np
import sys


def is_element_in_array(arr, e):
    '''
    Is and array 'e' (2 element vector) in array of arrays arr?
    
    arr is n+1 dimensional array (or None, in this case, False is returned),
    while e is n dismensional array
    searching 1000x1000 array - 1500 ms (it's 17 ms now :3 )
    '''
    # Probably as efficient as it could get with brute force
    if arr is None or arr.size == 0:
        return False
    a = arr.transpose()
    return (a[1][(a[0] == e[0])] == e[1]).any()


def get_n_random_positions(n, dims, restrictions):
    '''
    Get r random not repeated positions on board os size dims.
    Points in argument restrictions will be ommited.

    Efficiency increase -> If there is not a big chance the ranomly
    generated positions will hit not restricted points, generate possible
    positions and choose.
    '''
    # If requested occupied tiles could not fit onto board, ERROR
    restrictions_count = (len(restrictions) if not restrictions is None else 0)
    occupied_tiles = n + restrictions_count
    if occupied_tiles > np.prod(dims):
        print("ERROR: Too many agents and obstacles. Could not be fit onto the board.", sys.stderr)
        sys.exit(1)
    # Get the random positions now
    _positions = np.array([[x,y] for x in range(dims[0]) for y in range(dims[1]) \
                          if not is_element_in_array(restrictions, [x,y])])
    # If there is less possible positions then required positions, ERROR
    # Randomly choose some possible positions
    rng = np.random.default_rng()
    return rng.choice(_positions, n, replace=False)

def could_move(x, y, pos, dims):
    '''Will move from pos by x and y tiles still fit into board with dimensions dims?'''
    x_ok = not (pos[0] + x < 0 or pos[0] + x >= dims[0])
    y_ok = not (pos[1] + y < 0 or pos[1] + y >= dims[1])
    return x_ok and y_ok

def in_board(pos, dims):
    '''Is position inside the board?'''
    return (pos[0] >= 0 and pos[0] < dims[0]) and (pos[1] >= 0 and pos[1] < dims[1])
    
def could_see(x, y, agent_vision):
    '''Given the agent vision and position, will agent see tile x and y tiles away?'''
    could_see = (x**2+y**2)**(1/2) <= agent_vision
    return could_see
        
def could_discover(x, y, pos, dims, agent_vision):
    '''Determine if agent could discover a tile x and y tiles away.'''
    return could_move(x, y, pos, dims) and could_see(x, y, agent_vision)