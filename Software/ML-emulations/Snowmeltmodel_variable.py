from pcraster.framework import *
import numpy as np

#Initialise variables for later use
snow_array = np.empty(0)
precipitation_array = np.empty(0)
dem_array = np.empty(0)
temp_array = np.empty(0)
snowfall_array = np.empty(0)
rainfall_array = np.empty(0)
actualmelt_array = np.empty(0)
runoff_array = np.empty(0)
horizontal_pixels = 0
vertical_pixels = 0
timesteps = 0

class MyFirstModel(DynamicModel):
    def __init__(self):
        DynamicModel.__init__(self)
        setclone('dem.map')

    def initial(self):
        # ----------------------------------------------------------------------------#
        # Define initial conditions
        # ----------------------------------------------------------------------------#
        dem = self.readmap('dem')
        self.snow = 0.0
        elevationMeteoStation = 2058.1
        elevationAboveMeteoStation = dem - elevationMeteoStation

        global rate
        temperatureLapseRate = rate
        print(' \n Temperature lapse rate is: ', temperatureLapseRate, '\n')
        self.temperatureCorrection = elevationAboveMeteoStation * temperatureLapseRate

        # ----------------------------------------------------------------------------#
        # Prepare static data for ML read-out
        # ----------------------------------------------------------------------------#
        global horizontal_pixels
        global vertical_pixels
        global dem_array

        horizontal_pixels = len(pcr_as_numpy(dem)[1,:])
        vertical_pixels = len(pcr_as_numpy(dem)[:,1])
        demmap_array = pcr2numpy(dem, -1)
        demmap_array_flat = demmap_array.flatten()
        dem_array = np.append(dem_array, demmap_array_flat)
        # ----------------------------------------------------------------------------#

    def dynamic(self):
        # ----------------------------------------------------------------------------#
        # Calculate dynamic variables
        # ----------------------------------------------------------------------------#
        # Precipitation
        precipitation = timeinputscalar('precip.tss', 1)

        # Temperature
        temperatureObserved = timeinputscalar('temp.tss', 1)
        temperature = temperatureObserved - self.temperatureCorrection

        # Rainfall / Snowfall
        freezing = temperature < 0.0
        snowFall = ifthenelse(freezing, precipitation, 0.0)
        rainFall = ifthenelse(pcrnot(freezing), precipitation, 0.0)

        # Calculate snow -- Add fallen snow subtract melted
        self.snow = self.snow + snowFall
        potentialMelt = ifthenelse(pcrnot(freezing), temperature * 0.01, 0)
        actualMelt = min(self.snow, potentialMelt)
        self.snow = self.snow - actualMelt

        # Calculate generated Runoff
        runoffGenerated = actualMelt + rainFall

        # ----------------------------------------------------------------------------#
        # Report dynamic variables to maps
        # ----------------------------------------------------------------------------#
        # self.report(temperatureObserved, 'tempObs')
        # self.report(temperature, 'temp')
        # self.report(freezing, 'fr')
        # self.report(snowFall, 'snF')
        # self.report(rainFall, 'rF')
        # self.report(potentialMelt, 'pmelt')
        # self.report(actualMelt, 'amelt')
        # self.report(self.snow, 'snow')
        # self.report(runoffGenerated, 'rg')

        # ----------------------------------------------------------------------------#
        # Prepare dynamic drivers for ML read-out
        # ----------------------------------------------------------------------------#
        # Precipitation
        precipmap_array = pcr2numpy(precipitation, -1)
        precipmap_array_flat = precipmap_array.flatten()
        global precipitation_array
        precipitation_array = np.append(precipitation_array, precipmap_array_flat)

        # Temperature
        tempmap_array = pcr2numpy(temperature, -1)
        tempmap_array_flat = tempmap_array.flatten()
        global temp_array
        temp_array = np.append(temp_array, tempmap_array_flat)

        # Rainfall
        rnfllmap_array = pcr2numpy(rainFall, -1)
        rnfllmap_array_flat = rnfllmap_array.flatten()
        global rainfall_array
        rainfall_array = np.append(rainfall_array, rnfllmap_array_flat)

        # Snowfall
        snwfllmap_array = pcr2numpy(snowFall, -1)
        snwfllmap_array_flat = snwfllmap_array.flatten()
        global snowfall_array
        snowfall_array = np.append(snowfall_array, snwfllmap_array_flat)

        # Actual melt
        ameltmap_array = pcr2numpy(actualMelt, -1)
        ameltmap_array_flat = ameltmap_array.flatten()
        global actualmelt_array
        actualmelt_array = np.append(actualmelt_array, ameltmap_array_flat)

        # Run off
        runoffmap_array = pcr2numpy(runoffGenerated, -1)
        runoffmap_array_flat = runoffmap_array.flatten()
        global runoff_array
        runoff_array = np.append(runoff_array, runoffmap_array_flat)

        # Snow
        snowmap_array = pcr2numpy(self.snow, -1)
        snowmap_array_flat = snowmap_array.flatten()
        global snow_array
        snow_array = np.append(snow_array, snowmap_array_flat)
        # ----------------------------------------------------------------------------#


nrOfTimeSteps = 181
timesteps = nrOfTimeSteps


# temprate = [0.002, 0.003, 0.004, 0.005, 0.001]
# temprate = [0.001, 0.003, 0.004, 0.005, 0.002]
# temprate = [0.0035, 0.0036, 0.0037, 0.0038, 0.0039, 0.0041, 0.0042, 0.0043,0.0044, 0.0045, 0.004]
# temprate = [0.001, 0.002, 0.004, 0.005, 0.003]
# temprate = [0.001, 0.002, 0.003, 0.005, 0.004]
# temprate = [0.001, 0.002, 0.003, 0.004, 0.005]
temprate = [0.005, 0.005]
variable_list = temprate

for iii in range(len(variable_list)):
    rate = temprate[iii]
    myModel = MyFirstModel()
    dynamicModel = DynamicFramework(myModel, nrOfTimeSteps)
    dynamicModel.run()