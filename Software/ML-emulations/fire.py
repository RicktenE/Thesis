from pcraster import *
from pcraster.framework import *
import numpy as np

#Initialise variables for later use
fire_array = np.empty(0)
Area_fire_array = np.empty(0)
# precipitation_array = np.empty(0)
dem_array = np.empty(0)
# temp_array = np.empty(0)
# snowfall_array = np.empty(0)
# rainfall_array = np.empty(0)
# actualmelt_array = np.empty(0)
# runoff_array = np.empty(0)
horizontal_pixels = 0
vertical_pixels = 0
timesteps = 0

class Fire(DynamicModel):
  def __init__(self):
    DynamicModel.__init__(self)
    setclone('clone.map')

  def initial(self):
    # ----------------------------------------------------------------------------#
    # Define initial conditions
    # ----------------------------------------------------------------------------#

    #set initial fire map and change to scalar for later transformations
    self.fire = readmap('start.map')
    self.fire = scalar(self.fire)

    self.dem = self.readmap('firedem')
    self.gradient = slope(self.dem)
    self.report(self.gradient, 'gradient')

    # ----------------------------------------------------------------------------#
    # Prepare static data for ML read-out
    # ----------------------------------------------------------------------------#
    global horizontal_pixels
    global vertical_pixels
    global dem_array

    horizontal_pixels = len(pcr_as_numpy(self.dem)[1, :])
    vertical_pixels = len(pcr_as_numpy(self.dem)[:, 1])
    demmap_array = pcr2numpy(self.dem, -1)
    demmap_array_flat = demmap_array.flatten()
    dem_array = np.append(dem_array, demmap_array_flat)
  def dynamic(self):

    neighbours = window4total(self.fire)
    self.neighbourBurns = ifthenelse(neighbours > 0, boolean(1), boolean(0))
    # self.report(self.neighbourBurns, 'neighB')

    potentialNewFire = ifthenelse(self.neighbourBurns & ~boolean(self.fire) , boolean(1), boolean(0))

    fire_elevation = self.dem * self.fire
    PNF_elevation = scalar(potentialNewFire) * scalar(self.dem)

    # function part
    def elevation_probability(firemap, diagonal, shift, fire_elevation, PNF_elevation):
      import math as m
      F_neighbor_elevation = ifthenelse(firemap == 1, 0, shift0(fire_elevation, shift[0], shift[1]))

      if diagonal == True:
        angle = atan(((scalar(boolean(F_neighbor_elevation)) * PNF_elevation) - F_neighbor_elevation)*(-1) / m.sqrt(2))
      else:
        angle = atan(((scalar(boolean(F_neighbor_elevation)) * PNF_elevation) - F_neighbor_elevation)*(-1))

      angle_trans = ifthenelse(scalar(angle) > 180, scalar(angle) - 360, scalar(angle))
      prob_elev = 0.1*m.e ** (0.05 * scalar(angle_trans)) * scalar(boolean(F_neighbor_elevation))
      # prob_elev = 1*m.e ** (0.05 * scalar(angle_trans)) * scalar(boolean(F_neighbor_elevation))

      return prob_elev

    p_NW = elevation_probability(self.fire,  True, [1, 1] , fire_elevation, PNF_elevation)
    p_NE = elevation_probability(self.fire,  True, [1, -1], fire_elevation, PNF_elevation)
    p_SE = elevation_probability(self.fire,  True, [-1, -1], fire_elevation, PNF_elevation)
    p_SW = elevation_probability(self.fire,  True, [-1, 1], fire_elevation, PNF_elevation)
    p_N = elevation_probability(self.fire,  False, [1, 0], fire_elevation, PNF_elevation)
    p_E = elevation_probability(self.fire,  False, [0, -1], fire_elevation, PNF_elevation)
    p_S = elevation_probability(self.fire,  False, [-1, 0], fire_elevation, PNF_elevation)
    p_W = elevation_probability(self.fire,  False, [0, 1], fire_elevation, PNF_elevation)

    pElevation = p_N + p_S + p_W + p_E + p_NW + p_NE + p_SW + p_SE


    # p = 0.05  + pElevation
    p = self.gradient + pElevation
    # p = 0.25
    realization = scalar(uniform(1) <= p)

    NewFire = ifthenelse(boolean(potentialNewFire) & boolean(realization), boolean(1), boolean(0))
    # self.report(NewFire, 'FireEdge')

    # writing for ml application. Save entire map each timestep as a long flat array.
    # Where there are NaN's replace with -1 for visual representation in ML classification of 1's and 0's. if Nan's set to -9999
    # The rest of the values are not visible anymore
    firemap_array = pcr2numpy(self.fire, -1)
    firemap_array_flat = firemap_array.flatten()
    global fire_array
    fire_array = np.append(fire_array, firemap_array_flat)

    #just a check to see if it works properly
    # print('        ')
    # print(array_for_ml)
    self.purefire = ifthenelse(self.fire==1, self.fire, np.nan)

    # self.Area_fire = areaarea(ordinal(self.purefire))
    self.firemap_array = pcr2numpy(self.purefire, np.nan)
    self.firearea_array_flat = self.firemap_array.flatten()
    x = self.firearea_array_flat
    x = x[~numpy.isnan(x)]
    self.count_fire_pixels = len(x)
    global Area_fire_array
    Area_fire_array = np.append(Area_fire_array, self.count_fire_pixels)

    #update fire status
    self.fire = (self.fire + scalar(NewFire))
    # print('check')
    self.report(self.fire, 'fire')



nrOfTimeSteps=50
timesteps = nrOfTimeSteps
myModel = Fire()
dynamicModel = DynamicFramework(myModel,nrOfTimeSteps)
dynamicModel.run()

