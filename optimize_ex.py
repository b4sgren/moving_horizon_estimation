import numpy as np
import matplotlib.pyplot as plt
from car_animation import CarAnimation
from mhe import unwrap
import car_params as params
from copy import deepcopy


def generateVelocities(t):
    v = 1 + .5 * np.cos(2 * np.pi * 0.2 * t)
    w = -0.2 + 2 * np.cos(2 * np.pi * 0.6 * t)

    return v, w

def getMeasurements(state):
    z = np.zeros_like(params.lms, dtype=float)

    for i in range(z.shape[1]):
        lm = params.lms[:,i]
        ds = lm - state[0:2]

        r = np.sqrt(np.sum(ds**2))
        theta = np.arctan2(ds[1], ds[0]) - state[2]
        # theta = unwrap(theta) #not sure if this should be here or down a few lines

        z[0,i] = r + np.random.normal(0, params.sigma_r)
        z[1,i] = theta + np.random.normal(0, params.sigma_theta)
        z[1,i] = unwrap(z[1,i])

    return z

def propagateState(state, v, w, dt):
        theta = state[2]
        st = np.sin(theta)
        stw = np.sin(theta + w * dt)
        ct = np.cos(theta)
        ctw = np.cos(theta + w * dt)

        A = np.array([-v/w * st + v/w * stw,
                    v/w * ct - v/w * ctw,
                    w * dt])
        temp = state + A
        temp[2] = unwrap(temp[2])
        return temp

def optimize(mu, z):
    x0 = deepcopy(mu).flatten(order='F')
    mu = mu.flatten(order='F')

if __name__ == "__main__":
    dt = 0.1
    t = np.arange(0, 1, .1)
    vc, wc = generateVelocities(t)  # Control velocities
    v = vc + np.sqrt(params.alpha1 * vc**2 + params.alpha2 * wc**2) * np.random.randn(vc.size) # True velocities
    w = wc + np.sqrt(params.alpha3 * vc**2 + params.alpha4 * wc**2) * np.random.randn(wc.size)

    state = [np.zeros(3)]
    dead_reckon = [np.zeros(3)]
    mu = [np.zeros(3)]
    Sigma = [np.eye(3)] # Does the covariance shrink or do we just use the propagated covariance from odometry?
    zt = [getMeasurements(state[0])]

    #Generate the data
    for i in range(t.size):
        state.append(propagateState(state[-1], v[i], w[i], dt))
        zt.append(getMeasurements(state[-1]))
        mu.append(propagateState(mu[-1], vc[i], wc[i], dt))
        dead_reckon.append(propagateState(dead_reckon[-1], vc[i], wc[i], dt))
    
    #Put into numpy arrays
    state = np.array(state).T
    mu = np.array(mu).T
    dead_reckon = np.array(dead_reckon).T
    zt = np.array(zt).T

    optimize(mu, zt) # Need the first 10