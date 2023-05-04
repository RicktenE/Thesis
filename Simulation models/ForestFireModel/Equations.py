#Equations governing the evolution of a forest fire
import math as m
def pBurn (pH,pVeg,pDen,pW, pS):
    pBurn = pH*(1+pVeg)*(1+pDen)*pW*pS
    return pBurn

# pVeg : categories for the type of vegetation
# pDen : Density categories of the vegetation

# Wind
def pW(V, c1, c2,theta):
    fT = m.e**(V*c2*(m.cos(theta)-1))

    pW = m.e**(c1 * V)*fT
    return pW
# c1 and c2 constants to be determined
# theta angle between the direction of fire propagation and the direction of the wind
# V is wind speed --> with directions between 0 and 360 degrees??

# effect of ground elevation
def pS(l,E1,E2):
    """Effect of ground elevation
    pS = exp(a*thetas)
    l: length of cell sides
    E1, E2: elevation of cells
    a: costant"""
    a = 1 # --> constant adjusted from experimental data
    diagonal = 1
    cell =1
    if cell == diagonal:
        thetaS = m.atan((E1-E2)/ (l*m.sqrt(2))) # is in radians if we want degrees wrap atan in m.degrees
    else:
        thetaS = m.atan((E1 - E2) / l)  # is in radians if we want degrees wrap atan in m.degrees

    pS = m.e**(a*thetaS)

    return pS

# no spotting added yet