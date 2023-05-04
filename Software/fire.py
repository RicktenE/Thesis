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

    self.dem = readmap('dem.map')

    #get the number of pixels for the machine learning model
    global number_pixels
    number_pixels = len(pcr2numpy(self.dem, 0).flatten())

  def dynamic(self):

    neighbours = window4total(self.fire)
    self.neighbourBurns = ifthenelse(neighbours > 0, boolean(1), boolean(0))
    # self.report(self.neighbourBurns, 'neighB')

    potentialNewFire = ifthenelse(self.neighbourBurns & ~boolean(self.fire) , boolean(1), boolean(0))

    realization = uniform(1) < 0.35

    NewFire = ifthenelse(potentialNewFire & realization, boolean(1), boolean(0))
    # self.report(NewFire, 'FireEdge')

    # writing for ml application. Save entire map each timestep as a long flat array
    fire_array = pcr2numpy(self.fire, 0)
    flat_fire = fire_array.flatten()
    global array_for_ml
    array_for_ml = np.append(array_for_ml, flat_fire)

    #just a check to see if it works properly
    print('        ')
    print(array_for_ml)


    #update fire status
    self.fire = (self.fire + scalar(NewFire))
    print('check')
    self.report(self.fire, 'fire')



nrOfTimeSteps=50
myModel = Fire()
dynamicModel = DynamicFramework(myModel,nrOfTimeSteps)
dynamicModel.run()

