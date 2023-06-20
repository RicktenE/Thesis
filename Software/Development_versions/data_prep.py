import pandas as pd

def only_y_label(data, all_pixels_of_map, print_true=True, multiplesteps = True):
    """
    input: ndarray containing data
    input: Number of pixels of simulation.
    This function takes a flattened grid of pixels as input.
        When used for simulation models, the different timesteps are
        ammended below the previous timestep. This way each pixel has
        one row for each timestep. top nxm rows are timestep 1, last nxm
        rows are the last timestep. """

    import pandas as pd

    df = pd.DataFrame()

    # structure the timesteps
    df["x_input"] = pd.DataFrame(data)

    #Set results of timestep as label for previous timestep
    df["y_label"] = df["x_input"].shift(-(all_pixels_of_map))

    if not multiplesteps:
        #Remove the last timestep to avoid NaNs in label. The last simulated step does not have a new result. It is the last result
        #This result is the last
        df = df.iloc[:-(all_pixels_of_map)]

    if print_true:
        print('Total number of data points : ', len(df))
        # print('Length of one row of pixels, horizontal side of the grid: ', N)

    return df

def neighbour_as_feature(data, horizontal_pixels, vertical_pixels, print_true=True, multiplesteps = True):
    """
    input: ndarray containing data
    input: Number of pixels of simulation.
    This function takes a flattened grid of pixels as input.
        When used for simulation models, the different timesteps are
        ammended below the previous timestep. This way each pixel has
        one row for each timestep. top nxm rows are timestep 1, last nxm
        rows are the last timestep. """

    import pandas as pd

    # get the length of one row of pixels
    N = horizontal_pixels

    df = pd.DataFrame()

    # structure the timesteps
    df["x_input"] = pd.DataFrame(data)

    # Add the neighbours as predictive variables
    df["left"] = df["x_input"].shift(-1)
    df["left"].iloc[-1:] = df["x_input"].iloc[:1]  # fix nans

    df["top_left"] = df["x_input"].shift(-(N + 1))
    df["top_left"].iloc[-(N + 1):] = df["x_input"].iloc[:(N + 1)]  # fix nans

    df["top"] = df["x_input"].shift(-(N))
    df["top"].iloc[-(N):] = df["x_input"].iloc[:(N)]  # fix nans

    df["top_right"] = df["x_input"].shift(-(N - 1))
    df["top_right"].iloc[-(N - 1):] = df["x_input"].iloc[:(N - 1)]  # fix nans

    df["right"] = df["x_input"].shift(1)
    df["right"].iloc[:1] = df["x_input"].iloc[:1]  # fix nans

    df["bottom_right"] = df["x_input"].shift(N + 1)
    df["bottom_right"].iloc[:(N + 1)] = df["x_input"].iloc[:(N + 1)]  # fix nans

    df["bottom"] = df["x_input"].shift(N)
    df["bottom"].iloc[:(N)] = df["x_input"].iloc[:(N)]  # fix nans

    df["bottom_left"] = df["x_input"].shift(N - 1)
    df["bottom_left"].iloc[:(N - 1)] = df["x_input"].iloc[:(N - 1)]  # fix nans

    #Set results of timestep as label for previous timestep
    df["y_label"] = df["x_input"].shift(-(horizontal_pixels*vertical_pixels))



    if not multiplesteps:
        #Remove the last timestep to avoid NaNs in label. The last simulated step does not have a new result. It is the last result
        #This result is the last
        df = df.iloc[:-(horizontal_pixels*vertical_pixels)]

    if print_true:
        print('Total number of data points : ', len(df))
        print('Length of one row of pixels, horizontal side of the grid: ', N)

    return df

def driver_as_feature(df, driver, driver_name, all_pixels_of_map, multiplesteps = True, print_true = False):
    """
    input: data: Pandas.DataFrame, driver:np.array.flattened(), horizontal_pixels: int,vertical_pixels: int.
    output: adjusted Pandas.DataFrame
    This function adds the driver as a feature on the second last location """

    # In case this is not yet loaded
    import pandas as pd

    df = df
    name = driver_name
    row, col = df.shape

    if not multiplesteps:
        # print('check')
        # Remove last timestep
        driver = driver[:(-all_pixels_of_map)]


    # Insert driver on second last location (before the labels)
    secondlast = col-1
    df.insert(secondlast, name, driver)
    if print_true:
        print('Added ', name)
    return df

def VARIABLE_rate_as_feature(df, variable_rate, variable_rate_name, all_pixels_of_map, multiplesteps = True, print_true = False):
    """
    input: data: Pandas.DataFrame, driver:np.array.flattened(), horizontal_pixels: int,vertical_pixels: int.
    output: adjusted Pandas.DataFrame
    This function adds the driver as a feature on the second last location """

    # In case this is not yet loaded
    import pandas as pd

    df = df
    name = variable_rate_name
    row, col = df.shape



    # Insert driver on second last location (before the labels)
    secondlast = col-1
    df.insert(secondlast, name, variable_rate)
    if print_true:
        print('Added ', name)

    if not multiplesteps:
        # print('check')
        # Remove last timestep
        df = df[:(-all_pixels_of_map)]
        
    return df

def animate_data(data, steps, animation_name, vertical_pixels):
    '''Animation name must be a string ending in .mp4.
    Data is the data you want to emulate. Must be in form of a dataframe with rows of horizontal pixels
     columns of vertical pixels
     and the next timestep appended below the df.  '''

    # import libraries needed
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import imageio.v2 as imageio
    import matplotlib.cm as cm


    def save_plot_predict_rf(first_row, last_row, data, timestep):
        #subset the correct data
        data = data.iloc[first_row:last_row,:]



        # Create figure for animation
        fig, ax = plt.subplots(figsize=(5,5))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cmap = cm.gist_yarg
        # these outer values ( vmin, vmax) are set because otherwise the colorbar keeps shifting values with each new image in the animation
        im = ax.imshow(data, interpolation='nearest', cmap=cmap, vmin=0, vmax=0.275)
        ax.set_title('MSE timestep %i' %timestep)
        fig.colorbar(im,cax=cax, orientation='vertical', extend = 'both', ticks= [0.05,0.1,0.15,0.2,0.25])

        #save figure for animation
        plt.plot()
        plt.savefig(f'solution-{timestep}.png')
        plt.close()

    # last_row = 0
    first_row = 0

    # Create plots of simulation
    for timestep in range(steps):
        # timestep is 0,1,2,3,4,5...
        # first rows is 0+niks, 1+ last_row

        last_row = first_row + vertical_pixels
        save_plot_predict_rf(first_row = first_row, last_row= last_row, data=data, timestep= timestep)
        first_row = last_row

    with imageio.get_writer(animation_name, format='FFMPEG', mode='I', fps=3) as writer:
        for i in range(steps):
            image = imageio.imread(f'solution-{i}.png')
            writer.append_data(image)
