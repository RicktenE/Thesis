from pcraster import *
from pcraster.framework import *

class Fire(DynamicModel):
  def __init__(self):
    DynamicModel.__init__(self)
    setclone('clone.map')

  def initial(self):
    self.fire = readmap('start.map')
    self.fire = scalar(self.fire)

    self.dem = readmap('dem.map')


  def dynamic(self):

    neighbours = window4total(self.fire)
    self.neighbourBurns = ifthenelse(neighbours > 0, boolean(1), boolean(0))
    self.report(self.neighbourBurns, 'neighB')

    potentialNewFire = ifthenelse(self.neighbourBurns, self.fire+1, self.fire*0)

    realization = uniform(1)

    NewFire = potentialNewFire * realization

    #update fire status
    self.fire = (self.fire + NewFire)
    print('check')
    self.report(self.fire, 'fire')

nrOfTimeSteps=10
myModel = Fire()
dynamicModel = DynamicFramework(myModel,nrOfTimeSteps)
dynamicModel.run()

