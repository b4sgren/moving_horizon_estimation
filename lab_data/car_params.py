import numpy as np

#initial position
x0 = 2
y0 = -2
theta0 = 3 * np.pi / 4.0

#velocity motion model noise params
alpha1 = 0.1
alpha2 = 0.01
alpha3 = 0.01
alpha4 = 0.1

# Sensor noise params
sigma_r = .3 #m .1
sigma_theta = 0.02 #rad .01
