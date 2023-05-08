from pcraster import *
from pcraster.framework import *

class MyFirstModel(DynamicModel):
  def __init__(self):
    DynamicModel.__init__(self)
    setclone('dem.map')

  def initial(self):
    conversionValue = 3.0
    self.reservoir = 30.0 / conversionValue
    print('initial reservoir is: ', self.reservoir)

  def dynamic(self):
    outflow = 0.1 * self.reservoir 
    self.reservoir = self.reservoir - outflow 
    print(self.reservoir)

nrOfTimeSteps=10
myModel = MyFirstModel()
dynamicModel = DynamicFramework(myModel,nrOfTimeSteps)
dynamicModel.run()

  




