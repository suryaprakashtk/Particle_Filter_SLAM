import numpy as np
from map_functions import *
import matplotlib.pyplot as plt; plt.ion()
from time_sync import *
from motion_model_definition import *

path = "./result/dead_reckon/"

def dead_reckon_with_lidar(velocity,encoder_stamps,yaw_rate,imu_stamps,lidar,lidar_stamps,particles):

    [vel,ang_vel,lidar_value,time_value] = time_sync(velocity,encoder_stamps,yaw_rate,imu_stamps,lidar,lidar_stamps)

    dead = np.zeros(shape=(4,particles,time_value.shape[0]),dtype=float)
    lidar_particles = np.zeros(shape=(particles,3,lidar_value.shape[1]),dtype=float)
    velocity_input = np.mean(vel,axis=1)

    global_map = create_empty_map()
    log_odds = np.zeros(shape=(global_map['sizex'],global_map['sizey']),dtype=float)
    log_odds[:,:] = 0.5
    for i in range(1,time_value.shape[0]-2):
        tau = time_value[i] -  time_value[i-1]
        dead[0:3,:,i] = Exact_motion_model(dead[0:3,:,i-1],tau,velocity_input[i-1],ang_vel[i-1])


        rot_mat = np.zeros(shape=(particles,3,3),dtype=float)
        rot_mat[:,0,0] = np.cos(dead[2,:,i])
        rot_mat[:,0,1] = -np.sin(dead[2,:,i])
        rot_mat[:,1,0] = np.sin(dead[2,:,i])
        rot_mat[:,1,1] = np.cos(dead[2,:,i])
        rot_mat[:,2,2] = 1
        trans = np.zeros(shape=(particles,3,1),dtype=float)
        trans[:,0,0] = dead[0,:,i]
        trans[:,1,0] = dead[1,:,i]
        lidar_particles = rot_mat@lidar_value[:,:,i] + trans

        most_likely_partcile = 0

        origin = meters_to_pixel(dead[0,most_likely_partcile,i],dead[1,most_likely_partcile,i])
        local_occupied_cells = lidar_initialise(lidar_particles[most_likely_partcile,0:2,:],origin)

        for k in range(local_occupied_cells.shape[1]):
            output_my = my_bresenham2D(origin[0],origin[1],local_occupied_cells[0,k],local_occupied_cells[1,k])
            log_odds[output_my[0][-1],output_my[1][-1]] = log_odds[output_my[0][-1],output_my[1][-1]] + np.log(4)
            log_odds[output_my[0][:-1],output_my[1][:-1]] = log_odds[output_my[0][:-1],output_my[1][:-1]] - np.log(4)
        

        if(i%4890==0 and i>300):
            trajectory_pixel = meters_to_pixel(dead[0,0,0:i+1],dead[1,0,0:i+1])
            occu_grid = np.where(log_odds > 0.5, 0,np.where(log_odds < 0.5, 1, 0.5))
            fig3 = plt.figure()
            ax1 = fig3.add_subplot(111)
            ax1.imshow(occu_grid,cmap="gray")
            ax1.plot(trajectory_pixel[1,:],trajectory_pixel[0,:],'r',linewidth=1.5)
            plt.title("Dead Reckon Occupancy Grid Map for Data Set 2")
            
            name4 = "Deadreckon" + ".jpg"
            filename4 = path + name4
            plt.savefig(filename4,dpi=300, quality=50)
    return 

def dead_reckon_no_lidar(velocity,encoder_stamps,yaw_rate,imu_stamps,lidar,lidar_stamps,particles):

    [vel,ang_vel,lidar_value,time_value] = time_sync(velocity,encoder_stamps,yaw_rate,imu_stamps,lidar,lidar_stamps)
    
    dead = np.zeros(shape=(4,particles,time_value.shape[0]),dtype=float)
    velocity_input = np.mean(vel,axis=1)

    for i in range(1,time_value.shape[0]):
        tau = time_value[i] -  time_value[i-1]
        dead[0:3,:,i] = motion_model(dead[0:3,:,i-1],tau,velocity_input[i-1],ang_vel[i-1])

    fig3 = plt.figure()
    plt.plot(dead[0,:,:],dead[1,:,:])
    plt.title("Trajectory of 10 particle with Noise")
    plt.show
    name4 = "DS1_Noiseeffects" + ".jpg"
    filename4 = path + name4
    plt.savefig(filename4,dpi=300, quality=50)

    plt.close('all')

       
    return [dead[:,0,:],time_value]