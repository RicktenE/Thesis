from pcraster import *
from pcraster.framework import *
import numpy as np

array_for_ml = np.empty(0)
number_pixels = 0

class Fire(DynamicModel):
  def __init__(self):
    DynamicModel.__init__(self)
    setclone('clone.map')

  def initial(self):
    #set initial fire map and change to scalar for later transformations
    self.fire = readmap('start.map')
    self.fire = scalar(self.fire)

    self.dem = self.readmap('dem')
    self.gradient = slope(self.dem)
    self.report(self.gradient, 'gradient')

    #get the number of pixels for the machine learning model
    global number_pixels
    number_pixels = len(pcr2numpy(self.dem, 0).flatten())

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
      prob_elev = 0.001*m.e ** (0.05 * scalar(angle_trans)) * scalar(boolean(F_neighbor_elevation))
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
    fire_array = pcr2numpy(self.fire, -1)
    flat_fire = fire_array.flatten()
    global array_for_ml
    array_for_ml = np.append(array_for_ml, flat_fire)

    #just a check to see if it works properly
    # print('        ')
    # print(array_for_ml)


    #update fire status
    self.fire = (self.fire + scalar(NewFire))
    # print('check')
    self.report(self.fire, 'fire')



nrOfTimeSteps=100
myModel = Fire()
dynamicModel = DynamicFramework(myModel,nrOfTimeSteps)
dynamicModel.run()

