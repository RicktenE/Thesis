import pandas as pd
import numpy as np

# Run the model
import Snowmeltmodel

# Assign static features ---- array has length of 1x( all pixels )
dem = Snowmeltmodel.dem_array

# # Assign dynamic features ---- array has length of (timesteps)x( all pixels )
snow = Snowmeltmodel.snow_array

#Retrieve size of frame and timesteps
horizontal_pixels  = Snowmeltmodel.horizontal_pixels
vertical_pixels = Snowmeltmodel.vertical_pixels
timesteps = Snowmeltmodel.timesteps


