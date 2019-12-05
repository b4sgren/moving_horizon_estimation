import numpy as np

#initial position
x0 = -5
y0 = -3
theta0 = np.pi / 2.0

#velocity motion model noise params
alpha1 = 0.1
alpha2 = 0.01
alpha3 = 0.01
alpha4 = 0.1

# Sensor noise params
sigma_r = 0.1 #m
sigma_theta = 0.05 #rad
