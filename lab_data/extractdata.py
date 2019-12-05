from operator import itemgetter 
import scipy.io as sio

data = sio.loadmat("processed_data.mat")
truth_data = sio.loadmat("truth_data.mat")

landmarks, l_time, l_depth, l_bearing, odom_t, pos_odom_se2, vel_odom = \
itemgetter("landmarks", "l_time", "l_depth", "l_bearing", "odom_t", "pos_odom_se2", "vel_odom")(data)

landmarks = landmarks.T


t_truth, x_truth, y_truth, th_truth = \
itemgetter( "t_truth", "x_truth", "y_truth", "th_truth")(truth_data)
