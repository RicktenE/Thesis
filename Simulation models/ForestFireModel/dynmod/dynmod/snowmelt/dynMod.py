from pcraster import *
from pcraster.framework import *

class MyFirstModel(DynamicModel):
  def __init__(self):
    DynamicModel.__init__(self)
    setclone('dem.map')

  def initial(self):
    pass

  def dynamic(self):
    pass

nrOfTimeSteps=10
myModel = MyFirstModel()
dynamicModel = DynamicFramework(myModel,nrOfTimeSteps)
dynamicModel.run()

  




