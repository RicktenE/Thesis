from pcraster import *
from pcraster.framework import *
import numpy as np

array_for_ml = np.empty(0)
horizontal_pixels = 0
vertical_pixels = 0
timesteps = 0

class MyFirstModel(DynamicModel):
    def __init__(self):
        DynamicModel.__init__(self)
        setclone('dem.map')

    def initial(self):
        dem = self.readmap('dem')
        elevationMeteoStation = 2058.1
        elevationAboveMeteoStation = dem - elevationMeteoStation
        temperatureLapseRate = 0.005
        self.temperatureCorrection = elevationAboveMeteoStation * \
                                     temperatureLapseRate
        self.report(self.temperatureCorrection, 'tempCor')

        self.snow = 0.0

        global horizontal_pixels
        global vertical_pixels
        # number_pixels = len(pcr2numpy(dem, 0).flatten())
        horizontal_pixels = len(pcr_as_numpy(dem)[1,:])
        vertical_pixels = len(pcr_as_numpy(dem)[:,1])
    def dynamic(self):
        precipitation = timeinputscalar('precip.tss', 1)
        self.report(precipitation, 'pFromTss')
        temperatureObserved = timeinputscalar('temp.tss', 1)
        self.report(temperatureObserved, 'tempObs')

        temperature = temperatureObserved - self.temperatureCorrection
        self.report(temperature, 'temp')

        freezing = temperature < 0.0
        self.report(freezing, 'fr')
        snowFall = ifthenelse(freezing, precipitation, 0.0)
        self.report(snowFall, 'snF')
        rainFall = ifthenelse(pcrnot(freezing), precipitation, 0.0)
        self.report(rainFall, 'rF')

        self.snow = self.snow + snowFall

        potentialMelt = ifthenelse(pcrnot(freezing), temperature * 0.01, 0)
        self.report(potentialMelt, 'pmelt')
        actualMelt = min(self.snow, potentialMelt)
        self.report(actualMelt, 'amelt')

        self.snow = self.snow - actualMelt

        runoffGenerated = actualMelt + rainFall
        self.report(runoffGenerated, 'rg')

        # writing for ml application. Save entire map each timestep as a long flat array.
        # Where there are NaN's replace with -1 for visual representation in ML classification of 1's and 0's. if Nan's set to -9999
        # The rest of the values are not visible anymore

        # print(type(self.snow))

        snow_array = pcr2numpy(self.snow, -1)
        snow_runoff = snow_array.flatten()
        global array_for_ml
        array_for_ml = np.append(array_for_ml, snow_runoff)

        # just a check to see if it works properly
        # print('        ')
        # print(array_for_ml)

        self.report(self.snow, 'snow')


nrOfTimeSteps = 100
timesteps = nrOfTimeSteps
myModel = MyFirstModel()
dynamicModel = DynamicFramework(myModel, nrOfTimeSteps)
dynamicModel.run()