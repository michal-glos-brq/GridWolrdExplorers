import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import agent
import utils
import pprint

# Values for colors in matplotlib charts
OBSTACLE = 2
MAX_STANBY_TIME = 3


def print_agents_times():
    '''Print time consumed by measured methods'''
    pprint.pprint(agent.times)

class Board:
    '''
    Important properties:
    self.discovered: Matrix representing board (0: not discovered, 1: discovered)
    self.agents: Array of Agent instances
    self.p_obstacles: Array of points, where obstacles are located
    self.known_obstacles: Position of obstacles discovered
    self.moves: How could agents move (diagonally or not)
    self.agent_vision: Euclidean distance of agents sight
    self.finished:  Search finished
    '''
    def __init__(self, dims=(10,10), na=4, no=10, posa=None, poso=None,
                 move_diagonally=True, agent_type='random', agent_vision=1, animation_speed=200):
        '''
        Initialize Board class
        Input arguments:
            dims:               Tuple with size of initiated board (n,k)
            na:                 Number of desired agents (This or posa has to be defined)
            no:                 Number of desired obstacles (This or poso has to be defined)
            posa:               Array with positions of agents (priority over na)
            poso:               Array with positions of obstacles (priority over po)
            move_diagonally:    Boolean value if diagonal moves of agent are allowed
            agent_type:         Type of agent. Possible values are:
                random:    Move randomly on the board
            agent_vision:       Euclidean distance of how far could an agent see
        '''
        # Save important data
        self.dims = dims
        self.animation_speed = animation_speed
        # Determine sight distance of agents (has to see, where it goes)
        self.agent_vision = max(1.5,agent_vision) if move_diagonally else max(1,agent_vision)
        # Here will be stored positions discovered
        self.discovered = np.zeros(dims)
        # Generate or save agents and obstacles
        if poso is None:
            self.p_obstacles = utils.get_n_random_positions(no, dims, posa)
        else:
            self.p_obstacles = poso
        if posa is None:
            posa = utils.get_n_random_positions(na, dims, self.p_obstacles)
        # Create Agent instantions on positiones determined before
        self.agents = [agent.Agent(agent_type, pos, dims, move_diagonally, self.agent_vision) for pos in posa]
        
        # Discover on agents zero positions and intialize matplotlib animation
        self.discover()
        self._plot_init()
        # Search did not finish yet
        self.finished = False


    def get_boards(self):
        '''Get board with all information and board with discovered only information'''
        # Placeholder matrix
        gt_board = np.ones_like(self.discovered)
        # Place agents on the board
        for a in self.agents:
            gt_board[a.position[0], a.position[1]] = a.color
        # Place obstacles on board
        for o in self.p_obstacles:
            gt_board[o[0], o[1]] = OBSTACLE
        # Filter information for discovered only
        discovered_board = gt_board * self.discovered
        return gt_board, discovered_board


    def discover(self):
        '''Discover new areas on board'''
        # Itrate through each agent, determine it's sight area and update self.dicovered matrix
        for pos in self.agent_positions:
            # Itarate over each tile in square of side 2*agent_vision+1, crop it as a circle of agents sight
            for x in range(-int(self.agent_vision+0.5), int(self.agent_vision+0.5) + 1):
                for y in range(-int(self.agent_vision+0.5), int(self.agent_vision+0.5) + 1):
                    if utils.could_discover(x, y, pos, self.dims, self.agent_vision):
                        self.discovered[pos[0]+x,pos[1]+y] = 1
        # Compute all found obstacles
        self.known_obstacles = np.array([o for o in self.p_obstacles if self.discovered[o[0], o[1]]])
            
        
    def step(self):
        '''Perform one step for each agent'''
        # Iterate through each agent and it's position
        for agent in self.agents:
            if agent.standby_time < MAX_STANBY_TIME:
                # Pass known obstacles and agent positions in order not to bump into each other, update positions
                agent.step(self.known_obstacles, self.agents, self.discovered)
        # After iteration, update the discovered area and obstacles
        self.discover()

    ######################### Properties #########################

    @property
    def agent_positions(self):
        return [a.position for a in self.agents]

    ######################### Plotting part #########################

    def make_plot_nice(self):
        pass

    def make_plot_finished(self):
        pass

    def _plot_init(self):
        '''Initialize matplotlib figures and axes'''
        gt_board, discovered_board = self.get_boards()
        self.f, (self.a1, self.a2) = plt.subplots(1,2, figsize=(20,10))
        # Create color mapping and apply it to images
        self.cmap = matplotlib.colors.ListedColormap(['dimgray', 'honeydew', 'fuchsia', 'navy', 'green'])
        self.bounds= [0,1,2,3,4,5]
        self.norm = matplotlib.colors.BoundaryNorm(self.bounds, self.cmap.N)
        # The board with clear vision
        self.im1 = self.a1.imshow(gt_board, interpolation='none', cmap=self.cmap, norm=self.norm)
        # Tho board discovered by agents
        self.im2 = self.a2.imshow(discovered_board, interpolation='none', cmap=self.cmap, norm=self.norm)
        self.make_plot_nice()


    def plot_init(self):
        '''Initial values for animation'''
        # Initialize images
        gt, disc = self.get_boards()
        self.f.suptitle("Searching")
        self.ax1_text = self.a2.text(0.5, 1.035, f"Ground truth board", transform=self.a1.transAxes, ha="center")
        self.ax2_text = self.a2.text(0.5, 1.035, f"Discovered board: step 0.", transform=self.a2.transAxes, ha="center")
        self.im1.set_data(gt)
        self.im2.set_data(disc)
        return self.im1, self.im2, self.ax2_text,


    def re_plot(self, i):
        '''Perform one step of animation'''
        # Step on simulation, do not step if simulation over
        if self.discovered.all():
            if not self.finished:
                self.finished = True
                for a in self.agents:
                    a.finished()
                    self.f.suptitle("Finished")
                self.ax2_text.set_text(f"Discovered board: step {i}.")
        elif min([a.standby_time for a in self.agents]) >= MAX_STANBY_TIME:
            # Finished as 'Could not be finished'
            if not self.finished:
                self.finished = True
                for a in self.agents:
                    a.finished()
                    self.f.suptitle("Could not be finished")
                self.ax2_text.set_text(f"Discovered board: step {i}.")
        else:
            self.step()
            self.ax2_text.set_text(f"Discovered board: step {i+1}.")
        # Update images
        gt, disc = self.get_boards()
        self.im1.set_data(gt)
        self.im2.set_data(disc)
        return self.im1, self.im2, self.ax2_text,


    def search(self, frames=None, video_path=None, dpi=250):
        '''Start the agent search and animation, if required, save the video'''
        if video_path:
            writer = matplotlib.animation.FFMpegWriter(fps=24)
            anim = FuncAnimation(self.f, self.re_plot, init_func=self.plot_init, interval=self.animation_speed, blit=False, frames=frames)
            anim.save(video_path, writer=writer, dpi=dpi)
        else:
            anim = FuncAnimation(self.f, self.re_plot, init_func=self.plot_init, interval=self.animation_speed, blit=False)
            plt.show()        

    def debug(self):
        '''Debug search - no graphical output'''
        while True:
            self.step()