# Python code to implement Conway's Game Of Life
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os



# setting up the values for the grid
ON = 1
OFF = 0
vals = [ON, OFF]
df = pd.DataFrame()
flat_grid = np.empty(0)
flat_newgrid = np.empty(0)
number_pixels =np.empty(0)
count = 0
np.random.seed(1)

def randomGrid(N):
    """returns a grid of NxN random values"""
    return np.random.choice(vals, N * N, p=[0.2, 0.8]).reshape(N, N)


def addGlider(i, j, grid):
    """adds a glider with top left cell at (i, j)"""
    glider = np.array([[0, 0, 1],
                       [1, 0, 1],
                       [0, 1, 1]])
    grid[i:i + 3, j:j + 3] = glider

def update(frameNum, img, grid, N, max_frames, writetocsv):


    # copy grid since we require 8 neighbors
    # for calculation and we go line by line
    newGrid = grid.copy()

    global flat_grid
    global count

    print('Frame number %i' % count)

    for i in range(N):
        for j in range(N):
            # compute 8-neighbor sum
            # using toroidal boundary conditions - x and y wrap around
            # so that the simulation takes place on a toroidal surface.
            total = int((grid[i, (j - 1) % N] + grid[i, (j + 1) % N] +
                         grid[(i - 1) % N, j] + grid[(i + 1) % N, j] +
                         grid[(i - 1) % N, (j - 1) % N] + grid[(i - 1) % N, (j + 1) % N] +
                         grid[(i + 1) % N, (j - 1) % N] + grid[(i + 1) % N, (j + 1) % N]) / 1)

            # apply Conway's rules
            if grid[i, j] == ON:
                if (total < 2) or (total > 3):
                    newGrid[i, j] = OFF
            else:
                if total == 3:
                    newGrid[i, j] = ON



    # update data
    img.set_data(newGrid)
    grid[:] = newGrid[:]

    flat_grid = np.append(flat_grid, grid.flatten())
    if count == max_frames:
        if writetocsv:
            np.savetxt('GoL_flat.csv', flat_grid, delimiter= ',')

    count += 1
    return img


# main() function
def main():

    # Command line args are in sys.argv[1], sys.argv[2] ..
    # sys.argv[0] is the script name itself and can be ignored

    # parse arguments
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--grid-size', dest='N', required=False)
    parser.add_argument('--interval', dest='interval', required=False)
    parser.add_argument('--maxframes', dest='max_frames', required=False)
    parser.add_argument('--glider', action='store_true', required=False)
    parser.add_argument('--writetocsv', action='store_true', required=False)
    parser.add_argument('--mov-file', dest='movfile', required=False)
    args = parser.parse_args()

    # set grid size
    N = 50
    global number_pixels
    number_pixels = N
    if args.N and int(args.N) > 2:
        N = int(args.N)

    # set animation update interval
    updateInterval = 100
    if args.interval:
        updateInterval = int(args.interval)

    # declare grid
    grid = np.array([])

    # check if "glider" demo flag is specified
    if args.glider:
        grid = np.zeros(N * N).reshape(N, N)
        addGlider(1, 1, grid)

    else:  # populate grid with random on/off -
        # more off than on
        grid = randomGrid(N)

    # set up animation
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest')

    #Default number of maximum number of frames
    max_frames = 100

    if args.max_frames and int(args.max_frames) > 0:
        max_frames = int(args.max_frames)-1

    writetocsv = False
    if args.writetocsv:
        writetocsv = True

    # ax.set_title('Iteration %i' %frameNum)
    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, N, max_frames, writetocsv),
                                  frames=max_frames,
                                  interval=updateInterval,
                                  save_count=max_frames,
                                  repeat=False)
    if args.movfile:
        ani.save(args.movfile, fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()


# # call main
if __name__ == "__main__":
    main()
# for _ in range(10):
#     main()
