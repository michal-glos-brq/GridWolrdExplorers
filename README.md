### Project TileWorld:
This repository implements several autonomous searching algorithms.
 - Random - walks randomly
 - Naive - walks directly to the closes undiscovered spot on the map
 - Smart - evaluates the amount of tiles discovered per move needed - greedy (no planning, take what is the best now)
 - Smart & Coop - works as a Smart agent but penalizes discovering the same tiles by multiple agents

 ### How to run
 Script `python3 agensearch.py` could be run with different options:
  - `-c` opens the map editor before search (j: save and run, enter: run, left_mouse_click: change tile)
  - `-l` load map from JSON file 
  - `--diagonally` let agents move diagonally 
  - `--debug` set debug mode 
  - `-m` measures computational time 
  - `-d` int (square) or 2 (rectangle) ints, size of map 
  - `-a` agent count 
  - `-o` obstacle count 
  - `-v` agent vision radius 
  - `-t` agent type (naive, random, smart, smart_coop) 
  - `-s` save map as 
  - `-r` record video to the provided file
  - `--dpi` video dpi
  - `--animation-speed` video animation speed
  - `--frames` how many video frames to save
  - `-i` impossible map (small, medium, large)
