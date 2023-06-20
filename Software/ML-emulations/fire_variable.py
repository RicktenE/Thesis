from pcraster.framework import *
import numpy as np

#Initialise variables for later use



fire_array = np.empty(0)
Area_fire_array = np.empty(0)
dem_array = np.empty(0)
neighbours_array = np.empty(0)
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
    # self.report(self.gradient, 'gradient')

    # ----------------------------------------------------------------------------#
    # Prepare static data for ML read-out
    # ----------------------------------------------------------------------------#
    # global horizontal_pixels
    # global vertical_pixels
    global dem_array

    # horizontal_pixels = len(pcr_as_numpy(self.dem)[1, :])
    # vertical_pixels = len(pcr_as_numpy(self.dem)[:, 1])
    demmap_array = pcr2numpy(self.dem, -1)
    self.demmap_array_flat = demmap_array.flatten()
    self.demmap_array_flat = self.demmap_array_flat[self.demmap_array_flat != -1]
    dem_array = np.append(dem_array, self.demmap_array_flat)
  def dynamic(self):
    global neighbours_array
    neighbours = window4total(self.fire)

    self.neighbourBurns = ifthenelse(neighbours > 0, boolean(1), boolean(0))
    neighbourBurn_onmap = ifthenelse(self.dem != -1, scalar(self.neighbourBurns), scalar(self.neighbourBurns)*np.nan)
    neighboursmap_array = pcr2numpy(neighbourBurn_onmap, -1)
    neighboursmap_array = neighboursmap_array.flatten()
    neighboursmap_array = neighboursmap_array[neighboursmap_array != -1]
    neighbours_array = np.append(neighbours_array, neighboursmap_array)
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

      global rate
      global alpha_rate
      a = alpha_rate
      R = rate
      angle_trans = ifthenelse(scalar(angle) > 180, scalar(angle) - 360, scalar(angle))
      prob_elev = R*m.e ** (a * scalar(angle_trans)) * scalar(boolean(F_neighbor_elevation))
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
    firemap_array_flat = firemap_array_flat[firemap_array_flat != -1]
    count_fire_pixels = len(firemap_array_flat[firemap_array_flat == 1])

    # print('\n')
    # print('total_firemap_pixels: ', len(firemap_array_flat))
    # print('on_fire_pixels: ', count_fire_pixels)
    # print('dem_pixels: ', len(self.demmap_array_flat))

    global fire_array
    fire_array = np.append(fire_array, firemap_array_flat)

    global Area_fire_array
    Area_fire_array = np.append(Area_fire_array, count_fire_pixels)

    #update fire status
    self.fire = (self.fire + scalar(NewFire))
    # print('check')
    # self.report(self.fire, 'fire')

# alpha = [1.5E-3, 2E-3, 5E-3, 1.5E-2, 2E-2, 2.5E-2, 1E-2]
# alpha = [5E-3, 1E-2, 3E-1, 3.5E-1, 2E-1]
# R_ = [0.3,0.35,0.45,0.55, 0.5]
# R_ = [0.01]
# R_ = [1.5E-2, 2E-2,2.5E-2,3E-2,3.5E-2,4E-2,1E-2]
# R_ = [5E-3, 1.5E-1, 2.5E-1, 3.5E-1, 6E-1, 2E-1]
# R_ = [2E-1, 2.5E-1, 2.8E-1, 2.9E-1, 3.1E-1, 3.2E-1, 3.5E-1, 4E-1, 3E-1]
# R_ = [5E-3, 8E-3, 0.9E-2, 1.1E-2, 1.3E-2 2E-2, 2.5E-2, 1E-2]
# R_ = [1E-3, 1E-2, 1E-1, 2E-1, 3E-1]
# R_ = [0.002, 0.003, 0.004, 0.005, 0.001]
# R_ = [0.001, 0.002, 0.003, 0.004, 0.005]
# R_ = [2E-0, 4.5E-0, 4.7E-0, 5.2E-0, 5.5E-0, 7E-0, 5E-0]


#---------------interpolate R------------------
# alpha = [0.2]
# R_ = [5E-3, 1E-2, 1.5E-2, 2.5E-2, 3E-2, 3.5E-2, 2E-2]
# R_ = [5E-3, 1.5E-2, 2E-2, 2.5E-2, 3E-2, 3.5E-2, 1E-2]
#----------------interpolate Alpha-----------------
# alpha = [0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.2]
alpha = [0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.3]
R_ = [0.01]

variable_list = alpha
# variable_list = alpha


nrOfTimeSteps= 100
timesteps = nrOfTimeSteps

for iii in range(len(variable_list)):
    if variable_list == R_:
      rate = R_[iii] # variable of interest
      alpha_rate = alpha[0]  #Unchanged currently not variable of interest
      print(f' \n Alpha_{alpha}_R_{R_[iii]}\n')
    if variable_list == alpha:
      alpha_rate = alpha[iii] # Variable of interest
      rate = R_[0] # Unchanged currently not variable of interest
      print(f' \n Alpha_{alpha[iii]}_R_{R_}\n')
    myModel = Fire()
    dynamicModel = DynamicFramework(myModel, nrOfTimeSteps)
    dynamicModel.run()

# del fire_array
# del Area_fire_array
# del dem_array
# del horizontal_pixels
# del vertical_pixels
# del timesteps
# del rate
# del alpha_rate
