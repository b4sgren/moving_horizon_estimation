import numpy as np
import car_params as params
from collections import deque   # Using a deque makes poping from left and appending on right easy

def unwrap(phi):
    phi -= 2 * np.pi * np.floor((phi + np.pi) * 0.5/np.pi)
    return phi

class MHE:
    def __init__(self, t):
        self.dt = t
        self.Sigma = np.eye(3)

        self.pose_hist = deque()    # History of the last n poses
        self.Q_hist = deque()       # History of the noise from motion since it is dependent on v and w
        self.z_hist = deque()       # History of the measurements
        self.Sigma_hist = deque()   # History of the Pose Covariance

        self.N = 10  #Size of the window to optimize over

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
        G, V, M, Q = self.getJacobians(mu, v, w)

        mu_bar = self.propagateState(mu, v, w)
        Sigma_bar = G @ self.Sigma @ G.T + V @ M @ V.T

        # for i in range(z.shape[1]):
        #     lm = params.lms[:,i]
        #     ds = lm - mu_bar[0:2]
        #
        #     r = np.sqrt(ds @ ds)
        #     phi = np.arctan2(ds[1], ds[0]) - mu_bar[2]
        #     phi = unwrap(phi)
        #     z_hat = np.array([r, phi])
        #
        #     H = np.array([[-(lm[0] - mu_bar[0])/r, -(lm[1] - mu_bar[1])/r, 0],
        #                   [(lm[1] - mu_bar[1])/r**2, -(lm[0] - mu_bar[0])/r**2, -1]])
        #
        #     S = H @ Sigma_bar @ H.T + Q
        #     K = Sigma_bar @ H.T @ np.linalg.inv(S)
        #
        #     innov = z[:,i] - z_hat
        #     innov[1] = unwrap(innov[1])
        #     mu_bar = mu_bar + K @ (innov)
        #     mu_bar[2] = unwrap(mu_bar[2])
        #     Sigma_bar = (np.eye(3) - K @ H) @ Sigma_bar
        #
        # self.Sigma = Sigma_bar
        # mu_bar[2] = unwrap(mu_bar[2])
        return mu_bar, self.Sigma

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
        Q = np.diag([params.sigma_r**2, params.sigma_theta**2])

        return G, V, M, Q
