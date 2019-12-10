import numpy as np
import matplotlib.pyplot as plt
import car_params as params
import scipy.io as sio
from mhe2 import MHE
from mhe2 import unwrap
from extractdata import *


if __name__ == "__main__":
    mhe = MHE()

    x_hist = truth_data['x_truth']
    y_hist = truth_data['y_truth']
    theta_hist = truth_data['th_truth']
    t_truth = truth_data['t_truth']
    mu_hist = []
    err_hist = []
    covar_hist = []

    x0 = params.x0
    y0 = params.y0
    phi0 = params.theta0
    mu = np.array([x0, y0, phi0])
    Sigma = np.eye(3)

    t_prev = 0.0
    meas_index = 0
    for i in range(odom_t.size):
        #stuff for plotting
        mu_hist.append(mu)
        dt = odom_t.item(i) - t_prev
        t_prev = odom_t.item(i)

        j = np.argmin(np.abs(t_truth - odom_t.item(i))) + 1
        truth = np.array([x_truth.item(j), y_truth.item(j), th_truth.item(j)]) #Need to find the truth with the closest time to current time
        err = truth - mu
        err_hist.append(err)

        #find which measurements to use
        while l_time.item(meas_index) < odom_t.item(i):
            meas_index += 1
            if meas_index >= l_time.size:
                break
        meas_index -= 1

        lm_ind = np.argwhere(np.isfinite(l_depth[:,meas_index]))
        r = l_depth[lm_ind, meas_index]
        phi = l_bearing[lm_ind, meas_index]
        zt = np.hstack((r, phi)).T

        mu, Sigma = mhe.update(mu, zt, lm_ind, vel_odom[0,i], vel_odom[1,i], dt)
        covar_hist.append(np.diagonal(Sigma))

    fig1, ax1 = plt.subplots(nrows=3, ncols=1, sharex=True)
    mu_hist = np.array(mu_hist).T
    odom_t = odom_t.flatten()
    th_truth = unwrap(th_truth)
    ax1[0].plot(t_truth, x_truth, label="Truth")
    ax1[0].plot(odom_t, mu_hist[0,:], label="Est")
    ax1[0].set_ylabel("x (m)")
    ax1[0].legend()
    ax1[1].plot(t_truth, y_truth, label="Truth")
    ax1[1].plot(odom_t, mu_hist[1,:], label="Est")
    ax1[1].set_ylabel("y (m)")
    ax1[1].legend()
    ax1[2].plot(t_truth, th_truth, label="Truth")
    ax1[2].plot(odom_t, mu_hist[2,:], label="Est")
    ax1[2].set_xlabel("Time (s)")
    ax1[2].set_ylabel("$\psi$ (rad)")
    ax1[2].legend()
    ax1[0].set_title("Estimate vs Truth")
    
    plt.figure(2)
    plt.plot(x_truth, y_truth, 'b', label="Truth")
    plt.plot(mu_hist[0,:], mu_hist[1,:], 'r', label="Estimate")
    plt.scatter(landmarks[0,:], landmarks[1,:], marker='x', color='g', label="Landmarks")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    # plt.plot(pos_odom_se2[0,:], pos_odom_se2[1,:])
    
    err_hist = np.array(err_hist).T
    err_hist[2,:] = unwrap(err_hist[2,:])
    covar_hist = np.array(covar_hist).T
    sigma_hist = 2 * np.sqrt(covar_hist)
    fig2, ax2 = plt.subplots(nrows=3, ncols=1, sharex=True)
    ax2[0].plot(odom_t, err_hist[0,:], 'b', label="Error x")
    ax2[0].plot(odom_t, sigma_hist[0,:], 'r', label="2 $\sigma$ bound")
    ax2[0].plot(odom_t, -sigma_hist[0,:], 'r')
    ax2[0].legend()
    ax2[1].plot(odom_t, err_hist[1,:], 'b', label="Error y")
    ax2[1].plot(odom_t, sigma_hist[1,:], 'r', label="2 $\sigma$ bound")
    ax2[1].plot(odom_t, -sigma_hist[1,:], 'r')
    ax2[1].legend()
    ax2[2].plot(odom_t, err_hist[2,:], 'b', label="Error psi")
    ax2[2].plot(odom_t, sigma_hist[2,:], 'r', label="2 $\sigma$ bound")
    ax2[2].plot(odom_t, -sigma_hist[2,:], 'r')
    ax2[2].legend()
    
    plt.show()
    print("Finished")
    plt.close()
