from pcraster.framework import *
import math as m
import time
import numpy as np
start_time = time.time()

array_for_ml = np.empty(0)
number_pixels = 0

class RandomModel(DynamicModel):
  def __init__(self, a, R, c1, c2):
    DynamicModel.__init__(self)
    setclone('input/dem_clipped.map')
    self.a  = a
    self.R  = R
    self.c1 = c1
    self.c2 = c2


  def initial(self):
    # self.fire = readmap('input/old/start_12.map')
    # self.fire = readmap('results/start.map')

    global number_pixels
    number_pixels = len(pcr2numpy(self.fire, -99999).flatten())


    self.dem  = readmap('input/dem_clipped')
    veg = readmap('input/veg')
    # self.extend = readmap('input/extend_fire')
    self.extend = self.fire
    # self.extend = readmap('results/extendN')

    # make map with only ones
    self.ones = readmap('input/ones')

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

    pElevation = p_N + p_S + p_W + p_E + p_NW + p_NE + p_SW + p_SE

    self.report(pElevation, 'results/pe')

    # check if neighbouring cell is on fire then give value true including diagonal cells
    pfire = uniform(1)

    p = pElevation

    realization = scalar(pfire <= p)
    NewFire = potentialNewFire * realization

    firename = f"F{self.ap}{self.Rp}{self.c1p}{self.c2p}"
    self.report(self.fire, f'results/{firename}')

    #update the fire
    self.fire = (self.fire+ NewFire) #* self.extend




nrOfTimeSteps= 5
litw = [0.045, 0.131]

a_list  = [0.1]
R_list  = [100]
c1_list = [litw[0]*10]
c2_list = [litw[1]*5]



# c1*10, c2*5 beste
# c1*1 c2*50 slechtste

# A 0.05, R 0.1
for al in range(len(a_list)):
  a = [a_list[al], al+1]
  for Rl in range(len(R_list)):
    R = [R_list[Rl], Rl+1]
    for c1l in range(len(c1_list)):
      c1 = [c1_list[c1l], c1l +1]
      for c2l in range(len(c2_list)):
        c2 = [c2_list[c2l], c2l+1]
        myModel = RandomModel(a,R,c1,c2)
        dynamicModel = DynamicFramework(myModel, nrOfTimeSteps)
        dynamicModel.run()




# Save xcoordinates to csv file
# dfx = pd.DataFrame(myModel.xcoord)
# dfy = pd.DataFrame(myModel.ycoord)
# dfx.to_csv('xcoord')
# dfy.to_csv('ycoord')
print("\n\n--- %s seconds ---" % (time.time() - start_time))


