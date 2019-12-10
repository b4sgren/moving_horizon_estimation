import numpy as np
import car_params as params
from copy import deepcopy
from scipy.optimize import minimize

def unwrap(phi):
    phi -= 2 * np.pi * np.floor((phi + np.pi) * 0.5/np.pi)
    return phi

class MHE:
    def __init__(self, t):
        self.dt = t
        self.Sigma = np.eye(3)

        self.pose_hist = []    # History of the last n poses
        self.Q_hist = []       # History of the noise from motion since it is dependent on v and w. Is this needed
        self.z_hist = []       # History of the measurements
        self.Sigma_hist = []   # History of the Pose Covariance

        # self.pose_hist.append(np.zeros(3))
        # self.Sigma_hist.append(np.eye(3))

        self.N = 5  #Size of the window to optimize over

    def propagateState(self, state, v, w):
        theta = state[2]
        st = np.sin(theta)
        stw = np.sin(theta + w * self.dt)
        ct = np.cos(theta)
        ctw = np.cos(theta + w * self.dt)

        A = np.array([-v/w * st + v/w * stw,
                    v/w * ct - v/w * ctw,
                    w * self.dt])
        temp = state + A
        temp[2] = unwrap(temp[2])
        return temp

    def update(self, mu, z, v, w):
        G, V, M, R = self.getJacobians(mu, v, w)    # Motion wrt states, Motion wrt inputs, Process noise (v/w), Sensor noise
        Q = V @ M @ V.T

        mu_bar = self.propagateState(mu, v, w)
        self.Sigma = G @ self.Sigma @ G.T + Q

        self.pose_hist.append(mu_bar)
        self.Sigma_hist.append(self.Sigma)
        self.Q_hist.append(Q)
        self.z_hist.append(z)

        if len(self.pose_hist) >= self.N + 1: # + 1 is a hack so we have 10 measurements also
            mu = self.optimize(self.pose_hist[-self.N:], self.z_hist[-self.N:], self.Sigma_hist[-self.N:])
            mu = mu.reshape((3, int(mu.size/3)), order='F')
            mu[2] = unwrap(mu[2])
            mu_bar = mu[:,-1]
            for i in range(self.N):
                self.pose_hist[-(self.N - i)] = mu[:,i]
        else:
            mu = self.optimize(self.pose_hist, self.z_hist, self.Sigma_hist) # z_hist doesn't have the same length as pose_hist here b/c original position doesn't have
            mu = mu.reshape((3, int(mu.size/3)), order='F')
            mu[2] = unwrap(mu[2])
            mu_bar = mu[:,-1]
            for i in range(int(mu_bar.size/3)):
                self.pose_hist[i] = mu[:,i]

        return mu_bar, self.Sigma

    def optimize(self, mu, z, sigma):
        mu = np.array(mu).T.flatten(order='F')
        x0 = deepcopy(mu)
        z = np.swapaxes(np.array(z).T, 0, 1)
        sigma = np.array(sigma)

        x_hat_opt = minimize(self.objective_fun, mu, method='SLSQP', jac=False, args=(x0, z, sigma, params.lms), options={'ftol':1e-5, 'disp':False})

        return x_hat_opt.x

    def objective_fun(self, mu, x0, z, Sigmas, lms):
        e_x = 0
        e_z = 0
        e_x2 = 0
        R = np.diag([params.sigma_r**2, params.sigma_theta**2])
        R_inv = np.linalg.inv(R)
        z_hat = self.h(mu, lms)  #Get all expected measurements
        Omega = np.diag([3e3, 3e3, 1.5e3])
        Omega2= np.diag([1e3, 1e3, 0.5e3])

        dx = (x0 - mu).reshape((-1, 3, 1), order='F')
        dx[:,2] = unwrap(dx[:,2])
        # e_x = np.sum(dx.transpose(0,2,1) @ np.linalg.inv(Sigmas) @ dx) # Error between initialization and optimized
        e_x = np.sum(dx.transpose(0,2,1) @ Omega @ dx)

        temp = mu.reshape((-1,3,1), order='F')
        dx2 = np.diff(temp, axis=0)
        # dx2[:,2] = unwrap(dx2[:,2])
        temp2 = x0.reshape((-1,3,1),order='F')
        dx0 = np.diff(temp2, axis=0)
        # dx0[:,2] = unwrap(dx0[:,2])
        diff = dx0 - dx2
        diff[:,2] = unwrap(diff[:,2])
        e_x2 = np.sum(diff.transpose(0,2,1)@ Omega2 @ diff) #Punishes changing the original edge size

        dz = z - z_hat
        dz[1] = unwrap(dz[1])
        for i in range(z.shape[2]):
            e_z += np.sum(np.diagonal(dz[:,:,i].T @ R_inv @ dz[:,:,i]))

        return e_x + e_z + e_x2

    def h(self, mu, lms): #Need to check if this works
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

    def getJacobians(self, mu, v, w):
        theta = mu[2]
        ct = np.cos(theta)
        st = np.sin(theta)
        cwt = np.cos(theta + w * self.dt)
        swt = np.sin(theta + w * self.dt)

        #Jacobian of motion model wrt the states
        G = np.eye(3)
        G[0,2] = -v/w * ct + v/w * cwt
        G[1,2] = -v/w * st + v/w * swt

        #Jacobian of motion model wrt inputs
        V = np.array([[(-st + swt)/w, v * (st - swt)/w**2 + v * cwt * self.dt/w],
                      [(ct - cwt)/w, -v * (ct - cwt)/w**2 + v * swt * self.dt/w],
                      [0, self.dt]])

        #Process noise in motion model
        M = np.diag([params.alpha1 * v**2 + params.alpha2 * w**2,
                     params.alpha3 * v**2 + params.alpha4 * w**2])

        #Measurement Noise
        R = np.diag([params.sigma_r**2, params.sigma_theta**2])

        return G, V, M, R
