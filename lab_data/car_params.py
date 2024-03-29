import numpy as np

#initial position
x0 = 2.5
y0 = -1.7
theta0 = .175

#velocity motion model noise params
alpha1 = 0.1
alpha2 = 0.01
alpha3 = 0.01
alpha4 = 0.1

# Sensor noise params
sigma_r = .35 #m .3 
sigma_theta = 0.07 #rad .02
