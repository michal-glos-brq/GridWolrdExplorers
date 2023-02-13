#! /usr/bin/env python3
import board
import mapCreator
import numpy as np
import argparse
import json
import sys

## Hardcoded maps
###################################################################################################################################################

def wall(x,y,length, vertical=False):
    '''Build wall of obstacles'''
    if vertical:
        return np.array([[x, _y] for _y in range(y, y + length + 1)])
    else:
        return np.array([[_x, y] for _x in range(x, x + length + 1)])

def impossible(agent):
    dims = (128,128)
    posa = np.array([[8,8],[8,120],[120,8],[120,120],[80,48],[48,80],[48,48],[80,80]])
    poso = np.concatenate((wall(16,16,96), wall(16,112,96), wall(32,32,64), wall(32,96,64),
                      wall(16,16,96,vertical=True), wall(112,16,96,vertical=True), wall(32,32,64,vertical=True), wall(96,32,64,vertical=True)))
    poso = np.unique(poso, axis=0)
    agent_vision = 4.5
    return board.Board(dims=(128,128), posa=posa, poso=poso, agent_vision=agent_vision, agent_type=agent, move_diagonally=False, animation_speed=args.animation_speed)

def impossible_mid(agent):
    dims = (64,64)
    posa = np.array([[4,4], [60,60], [28,28], [36,36]])
    poso = np.concatenate((wall(8,8,48), wall(8,56,48), wall(16,16,32), wall(16,48,32),
                      wall(8,8,48,vertical=True), wall(48,8,48,vertical=True), wall(16,16,32,vertical=True), wall(48,16,32,vertical=True)))
    poso = np.unique(poso, axis=0)
    agent_vision = 3.5
    return board.Board(dims=(64,64), posa=posa, poso=poso, agent_vision=agent_vision, agent_type=agent, move_diagonally=False, animation_speed=args.animation_speed)

def impossible_small(agent):
    dims = (16,16)
    posa = np.array([[15,15], [0,0]])
    poso = np.concatenate((wall(4,12,8), wall(4,4,8),
                      wall(4,4,8,vertical=True), wall(12,4,8,vertical=True)))
    poso = np.unique(poso, axis=0)
    agent_vision = 2
    return board.Board(dims=dims, posa=posa, poso=poso, agent_vision=agent_vision, agent_type=agent, move_diagonally=False, animation_speed=args.animation_speed)

###################################################################################################################################################

def main(args):
    if len(args.dim) == 2:
        dims = tuple(args.dim)
    elif len(args.dim) == 1:
        dims = tuple(args.dim * 2)
    else:
        print("This is 2d board, please specify maximally 2 dimension with argument -d or --dim", file=sys.stderr)
        exit(69)

    if args.creative:
        c = mapCreator.CreativeBoard(args.save_path, dims)
        x = board.Board(dims=dims, na=len(c.posa), no=len(c.poso), posa=c.posa, poso=c.poso, agent_vision=args.vision, agent_type=args.agent_type, move_diagonally=args.diagonally, animation_speed=args.animation_speed)
    elif args.load_map:
        with open(args.load_map) as jsonFile:
            jsonObject = json.load(jsonFile)
            posa = jsonObject['posa']
            poso = jsonObject['poso']
            jsonFile.close()
            x = board.Board(dims=dims, na=len(posa), no=len(poso), posa=np.array(posa), poso=np.array(poso), agent_vision=args.vision, agent_type=args.agent_type, move_diagonally=args.diagonally, animation_speed=args.animation_speed)
    elif args.impossible:
        if args.impossible == 'small':
            x = impossible_small(args.agent_type)
        elif args.impossible == 'medium':
            x = impossible_mid(args.agent_type)
        elif args.impossible == 'large':
            x = impossible(args.agent_type)
        else:
            print("Please, specify impossible map size. {small, medium, large}")
    else:
        x = board.Board(dims=dims, na=args.agents, no=args.obstacles, agent_vision=args.vision, agent_type=args.agent_type, move_diagonally=args.diagonally, animation_speed=args.animation_speed)
    if args.debug:
        x.debug()
    else:
        x.search(frames=args.frames, video_path=args.video_path, dpi=args.dpi)
    if args.measure:
        board.print_agents_times()


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--creative', action='store_true',
    help="Open map editor.")
parser.add_argument('--diagonally', action='store_true',
    help="Allow agents to move diagonally.")
parser.add_argument('--debug', action='store_true',
    help="Trigger debug output.")
parser.add_argument('-m','--measure', action='store_true',
    help="Trigger measuring output.")
parser.add_argument('-d', '--dim', action='store', type=int, default=[20], nargs='+',
    help="Set dimension of the square map or 2 dimensions of rectangular map.")
parser.add_argument('-a', '--agents', action='store', type=int, default=6,
    help="Set number of agents.")
parser.add_argument('-o', '--obstacles', action='store', type=int, default=100,
    help="Set number of obstacles.")
parser.add_argument('-v', '--vision', action='store', type=int, default=2.3,
    help="Set vision of agents.")
parser.add_argument('-t', '--agent-type', action='store', type=str, default='smart', 
    help="Set type of agent behaviour. Could choose from {naive, random, smart, smart_coop}.")
parser.add_argument('-l', '--load-map', action='store', type=str, 
    help="Specify path to map you want to use.")
parser.add_argument('-s', '--save-path', action='store', type=str, default=None, 
    help="Specify path to save the map.")
parser.add_argument('-r', '--video-path', action='store', type=str, default='', 
    help="Store video of the animation on specified path.")
parser.add_argument('--dpi', action='store', type=int, default=250,
    help="DPI of output video.")
parser.add_argument('--animation-speed', '-f', action='store', type=int, default=250,
    help="Animation speed.")
parser.add_argument('--frames', action='store', type=int, default=1000,
    help="Frames in output video.")
parser.add_argument('-i', '--impossible', action='store', type=str, default='', 
    help="Store video of the animation on specified path. Choose from medium, small and large.")


if __name__ == "__main__":
    # Parse arguments, construct the LSTMctl class and execute required action
    args = parser.parse_args()
    main(args)