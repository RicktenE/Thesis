from pcraster import *
from pcraster.framework import *
import math as m
import time
import pandas as pd
import numpy as np
start_time = time.time()

array_for_ml = np.empty(0)
number_pixels = 0

class RandomModel(DynamicModel):
  def __init__(self, a, R, c1, c2, pb, ps,pg):
    DynamicModel.__init__(self)
    setclone('input/dem_clipped.map')
    self.a  = a[0]
    self.R  = R[0]
    self.c1 = c1[0]
    self.c2 = c2[0]
    self.pb = pb[0]
    self.ps = pb[0]
    self.pg = pb[0]
    self.ap  = a[1]
    self.Rp  = R[1]
    self.c1p = c1[1]
    self.c2p = c2[1]
    self.vegorder = pb[1]



  def initial(self):
    self.fire = readmap('input/startSharps')
    global number_pixels
    number_pixels = len(pcr2numpy(self.fire, -99999).flatten())
    # self.fire = readmap('results/begin')


    self.dem  = readmap('input/dem_clipped')
    veg = readmap('input/veg')
    # self.extend = readmap('input/extend_fire')
    self.extend = self.fire
    # self.extend = readmap('results/extendN')
    self.ones = readmap('input/ones')
    self.windSpeed = self.ones * 10

    # ycoord = ycoordinate(boolean(self.dem))
    # mapp = ifthenelse(ycoord >=43.584500, self.ones, np.nan )
    # # self.report(mapp, 'results/mapp')
    # fire_extend_reverse = ifthenelse(scalar(self.extend) == 1, self.ones*0, self.ones)
    # begin = ifthenelse(((scalar(ycoord) < 43.584500) & (scalar(fire_extend_reverse) == 1)), np.nan, scalar(fire_extend_reverse))
    # self.report(begin, 'results/begin')


    # self.extend_north = ifthenelse(ycoord >= 43.529064, self.ones, self.extend)
    # self.report(self.extend_north, 'results/extendN')
    # Get x and y coordinates of dem map (dem can be any map)
    # xcoord = xcoordinate(boolean(self.dem))
    # ycoord = ycoordinate(boolean(self.dem))
    # change xcoordinates to numpy array
    # self.xcoord = numpy_operations.pcr_as_numpy(xcoord)
    # self.ycoord = numpy_operations.pcr_as_numpy(ycoord)
    # find start point in dem map
    # start_point = ifthenelse((ycoord == 43.484715) & (xcoord == -114.121210) , self.dem/self.dem, self.dem*0)
    # self.report(start_point, 'input/startSharps')

    # make map with only ones
    # ones = ifthenelse(self.dem > 0, self.dem/self.dem, 0)
    # self.report(ones, 'input/ones')
    veg = nominal(veg)
    self.report(veg, 'veg')
    # self.pVeg = ifthenelse(veg == '1', veg*10, veg)
    # self.pVeg = ifthen(self.Pveg == 2, (self.Pveg/self.Pveg)*self.ps)
    # self.pVeg = ifthen(self.Pveg == 3, (self.Pveg/self.Pveg)*self.pg)
    # self.pVeg = ifthen(self.Pveg == 4, (self.Pveg/self.Pveg)*0)
    # self.report(self.Pveg, 'results/vegN')
    # self.pVveg = self.pVeg*0
    self.ph = 0.58

    # report(celllength(), 'results/area') #to find length of cell
    self.l = 0.00277178 # Cell length

  def dynamic(self):

    # potentialNewFire = (windowtotal(scalar(self.fire), 3 * self.l) > 0) & ~ self.fire
    potentialNewFire = scalar((windowtotal(self.fire, 3 * self.l) > 0)) * ifthenelse(self.fire == 1, self.ones*0, self.ones)
    fire_elevation = self.dem * self.fire
    PNF_elevation = potentialNewFire * self.dem

    # function part
    def elevation_probability(firemap, diagonal, shift, l, a, R, fire_elevation, PNF_elevation):
      F_neighbor_elevation = ifthenelse(firemap == 1, 0, shift0(fire_elevation, shift[0], shift[1]))

      if diagonal == True:
        angle = atan(((scalar(boolean(F_neighbor_elevation)) * PNF_elevation) - F_neighbor_elevation)*(-1) / (l * m.sqrt(2)))
      else:
        angle = atan(((scalar(boolean(F_neighbor_elevation)) * PNF_elevation) - F_neighbor_elevation)*(-1) / (l))

      angle_trans = ifthenelse(scalar(angle) > 180, scalar(angle) - 360, scalar(angle))
      prob_elev = R * m.e ** (a * scalar(angle_trans)) * scalar(boolean(F_neighbor_elevation))
      return prob_elev

    def wind_probability( fire, shift, angle):
      F_neighbor = ifthenelse(fire == 1, 0, shift0(fire, shift[0], shift[1]))

      windSpeed = self.windSpeed * scalar(F_neighbor)
      angle *= -1
      f = m.e**(windSpeed*self.c2 * (angle - 1)) * scalar(F_neighbor)

      prob_wind = m.e**(self.c1*windSpeed)*f

      return prob_wind


    # Function calls
    pw_N  = wind_probability( self.fire, [1,0]   , 0)
    pw_E  = wind_probability( self.fire, [0, -1] , 1)
    pw_S  = wind_probability( self.fire, [-1, 0] , 0)
    pw_W  = wind_probability( self.fire, [0, 1]  , -1)
    pw_NE = wind_probability( self.fire, [1, -1] , m.sqrt(2)/2)
    pw_SE = wind_probability( self.fire, [-1, -1], m.sqrt(2)/2)
    pw_SW = wind_probability( self.fire, [-1, 1] , -m.sqrt(2)/2)
    pw_NW = wind_probability( self.fire, [1, 1]  ,-m.sqrt(2)/2)


    p_NW = elevation_probability(self.fire,  True, [1, 1] , self.l, self.a, self.R\
                                 , fire_elevation, PNF_elevation)
    p_NE = elevation_probability(self.fire,  True, [1, -1], self.l, self.a, self.R\
                                 , fire_elevation, PNF_elevation)
    p_SE = elevation_probability(self.fire,  True, [-1, -1], self.l, self.a, self.R\
                                 , fire_elevation, PNF_elevation)
    p_SW = elevation_probability(self.fire,  True, [-1, 1], self.l, self.a, self.R\
                                 , fire_elevation, PNF_elevation)
    p_N = elevation_probability(self.fire,  False, [1, 0], self.l, self.a, self.R\
                                , fire_elevation, PNF_elevation)
    p_E = elevation_probability(self.fire,  False, [0, -1], self.l, self.a, self.R\
                                , fire_elevation, PNF_elevation)
    p_S = elevation_probability(self.fire,  False, [-1, 0], self.l, self.a, self.R\
                                , fire_elevation, PNF_elevation)
    p_W = elevation_probability(self.fire,  False, [0, 1], self.l, self.a, self.R\
                                , fire_elevation, PNF_elevation)

    # Adding probabilities
    pWind = pw_N + pw_E + pw_S + pw_W + pw_NE + pw_SE + pw_SW + pw_NW
    pElevation = p_N + p_S + p_W + p_E +p_NW + p_NE + p_SW + p_SE

    self.report(pWind, 'results/pw')
    self.report(pElevation, 'results/pe')
    # check if neighbouring cell is on fire then give value true including diagonal cells
    pfire = uniform(1)
    # p = self.ph*(1+self.pVeg)  * pElevation
    p = self.ph*(1+0)  * pElevation
    realization = scalar(pfire <= p)
    NewFire = potentialNewFire * realization

    firename = f"F{self.ap}{self.Rp}{self.c1p}{self.c2p}"
    self.report(self.fire, f'results/{firename}')

    fire_array = pcr2numpy(self.fire, -999999)
    flat_fire = fire_array.flatten()
    global array_for_ml
    array_for_ml = np.append(array_for_ml, flat_fire )
    self.fire = (self.fire+ NewFire) #* self.extend
    # self.fire = pcror(self.fire, NewFire) *self.extend
    print('        ')
    print(array_for_ml)




nrOfTimeSteps= 50
litw = [0.045, 0.131]

a_list  = [0.05]
R_list  = [0.1]
c1_list = [litw[0]*10]
c2_list = [litw[1]*5]

pb_list = [0.1]
ps_list = [0.1]
pg_list = [0.9]


# c1*10, c2*5 beste
# c1*1 c2*50 slechtste

# A 0.05, R 0.1
for v in range(len(pb_list)):
  pb = [pb_list[v], v+1]
  ps = [ps_list[v], v+1]
  pg = [pg_list[v], v+1]
  for al in range(len(a_list)):
    a = [a_list[al], al+1]
    for Rl in range(len(R_list)):
      R = [R_list[Rl], Rl+1]
      for c1l in range(len(c1_list)):
        c1 = [c1_list[c1l], c1l +1]
        for c2l in range(len(c2_list)):
          c2 = [c2_list[c2l], c2l+1]
          myModel = RandomModel(a,R,c1,c2, pb,ps, pg)
          dynamicModel = DynamicFramework(myModel, nrOfTimeSteps)
          dynamicModel.run()




# Save xcoordinates to csv file
# dfx = pd.DataFrame(myModel.xcoord)
# dfy = pd.DataFrame(myModel.ycoord)
# dfx.to_csv('xcoord')
# dfy.to_csv('ycoord')
print("\n\n--- %s seconds ---" % (time.time() - start_time))


