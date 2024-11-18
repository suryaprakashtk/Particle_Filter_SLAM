import numpy as np
import matplotlib.pyplot as plt; plt.ion()


def lidar_to_cart(lidar_angle_min,lidar_angle_max,lidar_angle_increment,lidar_range_min,lidar_range_max,lidar_ranges,lidar_stamsp):
    no_timesteps = np.shape(lidar_stamsp)[0]
    no_ranges_per_scan = np.shape(lidar_ranges)[0]
    coordinates = np.zeros(shape=(3,no_ranges_per_scan,no_timesteps), dtype=float)
    # Alpha angle sequences for 1 scan
    # perturbation of less than angle increment is added to include the last scan data.
    alpha = np.arange(lidar_angle_min, lidar_angle_max+0.000001, lidar_angle_increment[0,0])
    x_cos = np.cos(alpha)
    x_sin = np.sin(alpha)
    # Making ranges value to 0 if its too far or too near
    lidar_ranges[lidar_ranges>=lidar_range_max] = 0.0
    lidar_ranges[lidar_ranges<=lidar_range_min] = 0.0
    # calculating x,y,z for each time stamps
    for i in range(no_timesteps):
        # x cooridnates
        coordinates[0,:,i] = np.multiply(lidar_ranges[:,i],x_cos)
        # y coordinates
        coordinates[1,:,i] = np.multiply(lidar_ranges[:,i],x_sin)
        # z coordinates are zeroes by default
    print("lidar_to_cart function from lidar.py executed")
    # return a 3d array of points with x,y,z in 2d and thrid dimensions as the number of timesteps of lidar scan
    return coordinates


def lidar_to_body(lidar_frame):
    body_frame = np.zeros(shape=lidar_frame.shape,dtype=float)
    no_timesteps = lidar_frame.shape[2]
    # lidar frame is in same orientation as body frame
    # lidar frame is at a offset of p_lidar_to_body in terms of position
    # Origin at the center between the axles and 20 mm above the axle plane
    rotation_lidar_to_body = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=float)
    p_lidar_to_body = np.array([[0.133,0,0.494]],dtype=float)
    for i in range(no_timesteps):
        body_frame[:,:,i] = np.matmul(rotation_lidar_to_body,lidar_frame[:,:,i]) + p_lidar_to_body.T
    print("lidar_to_body function from lidar.py executed")
    return body_frame

