def neighbour_as_feature(data, horizontal_pixels, vertical_pixels):
    """
    input: ndarray containing data
    input: Number of pixels of simulation.
    This function takes a flattened grid of pixels as input.
        When used for simulation models, the different timesteps are
        ammended below the previous timestep. This way each pixel has
        one row for each timestep. top nxm rows are timestep 1, last nxm
        rows are the last timestep. """

    import pandas as pd
    import numpy as np

    N = horizontal_pixels #get the length of one row of pixels
    data = data

    print('Total number of data points : ', len(data))
    print('Length of one row of pixels, horizontal side of the grid: ', N)

    df = pd.DataFrame()

    #structure the timesteps
    df["x_input"] = pd.DataFrame(data)

    # Add the neighbours as predictive variables
    df["left"] = df["x_input"].shift(-1)
    df["left"].iloc[-1:] = df["x_input"].iloc[:1]  # fix nans

    df["top_left"] = df["x_input"].shift(-(N + 1))
    # df["top_left"] = df["x_input"].shift(-6)
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

    #Remove the last timestep to avoid NaNs in label. The last simulated step does not have a new result. It is the last result
    #This result is the last
    df = df.iloc[:-(horizontal_pixels*vertical_pixels)]

    #### !! ###
    # Deal with the edges somehow

    #--!! Ratio of Area/edges decreases by 4x/(x^2) for a square grid. therefore the larger the simulation the smaller the
    # error due to the wrong neighbour classification at the edges.

    # Perhaps check how much of an influence this hold and from when it becomes irrelevent

    ### !! ###

    return df
