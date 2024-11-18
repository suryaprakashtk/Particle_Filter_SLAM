import numpy as np
from encoder import *
from lidar import *
from utils import *
from particle_filter import *
from dead_reckon import *
from texture import *

path = "./data/"


if __name__ == '__main__':
  dataset = 20
  
  with np.load(path + "Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

  with np.load(path +"Hokuyo%d.npz"%dataset) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamsp = data["time_stamps"]  # acquisition times of the lidar scans
    
  with np.load(path + "Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
  
  with np.load(path + "Kinect%d.npz"%dataset) as data:
    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

# measuring time referenced from the minimum timestamp of all 4 data
t0 = min(encoder_stamps[0],lidar_stamsp[0],imu_stamps[0],disp_stamps[0],rgb_stamps[0])
encoder_time =  encoder_stamps - t0
lidar_time = lidar_stamsp - t0
imu_time = imu_stamps - t0
disp_time = disp_stamps - t0
rgb_time = rgb_stamps - t0


# Dead reckoning
# Average right-left distances and velocities travelled over time
[distance_rl,velocity_rl] = encode_data_manipulation(encoder_counts,encoder_time)

# Lidar sensor frame coordiantes. In 3 x 1081 x no of lidar timesteps. In X,Y,Z format
lidar_sensor_coordiantes = lidar_to_cart(lidar_angle_min,lidar_angle_max,lidar_angle_increment,lidar_range_min,lidar_range_max,lidar_ranges,lidar_stamsp)
# Lidar values in robot body frame coordiantes. In 3 x 1081 x no of lidar timesteps. In X,Y,Z format
lidar_body_coordiantes = lidar_to_body(lidar_sensor_coordiantes)

no_of_partciles = 100

[dead_trajectory,timevalue] = dead_reckon_no_lidar(velocity_rl,encoder_time,imu_angular_velocity[2,:],imu_time,lidar_body_coordiantes,lidar_time,no_of_partciles)
dead_reckon_with_lidar(velocity_rl,encoder_time,imu_angular_velocity[2,:],imu_time,lidar_body_coordiantes,lidar_time,no_of_partciles)
[best_path,grid] = particle_filter(velocity_rl,encoder_time,imu_angular_velocity[2,:],imu_time,lidar_body_coordiantes,lidar_time,no_of_partciles)
texture_plot = texture_map(encoder_time,rgb_time,disp_time,best_path,timevalue)


