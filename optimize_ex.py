import numpy as np
import matplotlib.pyplot as plt
from car_animation import CarAnimation
from mhe import unwrap
import car_params as params
from copy import deepcopy
from scipy.optimize import minimize


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

        z[0,i] = r #+ np.random.normal(0, params.sigma_r)
        z[1,i] = theta #+ np.random.normal(0, params.sigma_theta)
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

def optimize(mu, z, lms):
    x0 = deepcopy(mu).flatten(order='F')
    mu = mu.flatten(order='F')

    x_hat_opt = minimize(objective_fun, mu, method='SLSQP', jac=False, args=(x0, z, lms), options={'ftol':1e-5, 'disp':True})

    print("Optimized: ", x_hat_opt)
    print("Original: ", x0)

    return x_hat_opt.x

def objective_fun(mu, x0, z, lms):
    R = np.diag([params.sigma_r**2, params.sigma_theta**2])
    R_inv = np.linalg.inv(R)
    z_hat = h(mu, lms)  #Get all expected measurements

    Sigma = np.eye(3) * 0.1 # haven't propagated covariance. Will use this for now
    Omega = np.linalg.inv(Sigma)

    dx = (mu - x0).reshape((3, int(mu.size/3)), order='F')
    e_x = np.sum(np.diagonal(dx.T @ Omega @ dx))

    dz = z - z_hat
    e_z = 0.0
    for i in range(z.shape[2]):
        e_z += np.sum(np.diagonal(dz[:,:,i].T @ R_inv @ dz[:,:,i])) # Cant do this multiplication

    return e_x + e_z

def h(mu, lms): #Need to check if this works
    z_hat = np.zeros((2, lms.shape[1], int(mu.size/3.0)))
    mu_temp = mu.reshape((3, int(mu.size/3)), order='F')
    for i in range(lms.shape[1]):
        lm = lms[:,i]
        ds = lm.reshape((2,1)) - mu_temp[0:2]
        r = np.sqrt(np.sum(ds * ds, axis=0))
        theta = np.arctan2(ds[1], ds[0]) - mu_temp[2]
        theta = unwrap(theta)

        z_temp = np.vstack((r, theta))
        z_hat[:,i,:] = z_temp

    return z_hat


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
    zt = np.array(zt)
    zt = np.swapaxes(zt.T, 0, 1)    #Put in the correct shape for subtraction

    mu = optimize(mu, zt, params.lms)
    mu = mu.reshape((3, int(mu.size/3)), order='F')

    plt.figure(1)
    plt.plot(state[0,:], state[1,:], 'b')
    plt.plot(dead_reckon[0,:], dead_reckon[1,:], 'k')
    plt.plot(mu[0,:], mu[1,:], 'r')

    plt.show()
