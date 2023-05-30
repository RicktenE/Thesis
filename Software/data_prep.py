import pandas as pd


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

def driver_as_feature(df, driver, driver_name, horizontal_pixels, vertical_pixels, multiplesteps = True, print_true = False):
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
        driver = driver[:(-horizontal_pixels*vertical_pixels)]


    # Insert driver on second last location (before the labels)
    secondlast = col-1
    df.insert(secondlast, name, driver)
    if print_true:
        print('Added ', name)
    return df




