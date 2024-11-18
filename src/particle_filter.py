import numpy as np
from utils import *
from map_functions import *
import matplotlib.pyplot as plt; plt.ion()
from time_sync import *
from motion_model_definition import *

path = "./result/SLAM/"

def resample(data,samples):
    res = np.zeros(shape = data.shape,dtype=float)
    probability_density = data[3,:]
    probability_density /= np.sum(probability_density)
    index = np.arange(0,samples,1)
    random_index = np.random.choice(index, size = samples, p=probability_density)
    res = data[:,random_index]
    res[3,:] = 1/samples
    return res

# position_hypothesis = np.zeros(shape=(4,7),dtype=float)
# position_hypothesis[0,:] = np.array([0.5,1.5,2.5,3.6,4.6,5.7,6.8])
# position_hypothesis[1,:] = np.array([-0.2,-1.2,-2.3,-3.4,-4.4,-5.5,-6.6])
# position_hypothesis[2,:] = np.array([0.001,0.003,0.007,0.05,0.9,0.8,0.687])
# particles = 7
# position_hypothesis[3,:] = np.array([1,1,1,2,2,2,10])
# position_hypothesis = resample(position_hypothesis,particles)

# print("halt")

def correlation_vector(grid,log_odds ,x_im, y_im, vp, xs, ys):
    # im = np.where(log_odds > 0, 1,np.where(log_odds < 0, -1, 0))
    im = grid
    no_part = vp.shape[2]
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = x_im[0]
    xmax = x_im[-1]
    xresolution = (xmax-xmin)/(nx-1)
    ymin = y_im[0]
    ymax = y_im[-1]
    yresolution = (ymax-ymin)/(ny-1)
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys,vp.shape[2]))
    x_range = np.zeros(shape=(nxs*nys),dtype=float)
    y_range = np.zeros(shape=(nxs*nys),dtype=float)
    k=0
    for i in range(nxs):
        for j in range(nys):
            x_range[k] = xs[j]
            y_range[k] = ys[i]
            k = k + 1
    local_points_with_perturb_meters = np.zeros(shape= (vp.shape[0],vp.shape[1],vp.shape[2],nxs*nys),dtype=float)
    local_points_with_perturb_meters[0,:,:,:] = vp[0,:,:,np.newaxis] + x_range
    local_points_with_perturb_meters[1,:,:,:] = vp[1,:,:,np.newaxis] + y_range
    local_points_with_perturb = np.zeros(shape= (vp.shape[0],vp.shape[1],vp.shape[2],nxs*nys),dtype=int)
    local_points_with_perturb[0,:,:,:] = np.int16(np.round((local_points_with_perturb_meters[0,:,:,:]-xmin)/xresolution))
    local_points_with_perturb[1,:,:,:] = np.int16(np.round((local_points_with_perturb_meters[1,:,:,:]-ymin)/yresolution))
    valid = np.zeros(shape= (vp.shape[0],vp.shape[1],vp.shape[2],nxs*nys),dtype=float)
    valid[0,:,:,:] = np.logical_and((local_points_with_perturb[0,:,:,:] >=0), (local_points_with_perturb[0,:,:,:] < nx))
    valid[1,:,:,:] = np.logical_and((local_points_with_perturb[1,:,:,:] >=0), (local_points_with_perturb[1,:,:,:] < ny))
    valid_index = np.logical_and((valid[0,:,:,:] ==True), (valid[1,:,:,:] == True))
    # making 0,0 pixels for all out of bound values
    lidar_end_pts_transpose = local_points_with_perturb.T
    lidar_end_pts_transpose[lidar_end_pts_transpose[:,:,:,0]<0,:] = 0
    lidar_end_pts_transpose[lidar_end_pts_transpose[:,:,:,0]>=nx,:] = 0
    lidar_end_pts_transpose[lidar_end_pts_transpose[:,:,:,1]<0,:] = 0
    lidar_end_pts_transpose[lidar_end_pts_transpose[:,:,:,1]>=ny,:] = 0
    output = im[lidar_end_pts_transpose[:,:,:,0],lidar_end_pts_transpose[:,:,:,1]].T
    output[valid_index==False]=0
    corr = np.sum(output,axis=0)

    arr_3d = corr.reshape((corr.shape[0], nxs, nys))
    final = arr_3d.T
    cpr = correlation_scaling(final)
    return cpr

def my_correlation2(grid,log_odds, x_im, y_im, vp, xs, ys):
    # im = np.where(log_odds > 0, 1,np.where(log_odds < 0, -1, 0))
    # im = grid
    no_part = vp.shape[0]
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = x_im[0]
    xmax = x_im[-1]
    xresolution = (xmax-xmin)/(nx-1)
    ymin = y_im[0]
    ymax = y_im[-1]
    yresolution = (ymax-ymin)/(ny-1)
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys,no_part))
    # cpr2 = np.zeros((nxs, nys,no_part))
    for a in range(no_part):
        for jy in range(0,nys):
            y1 = vp[a,1,:] + ys[jy] # 1 x 1076
            iy = np.int16(np.round((y1-ymin)/yresolution)) -1 
            for jx in range(0,nxs):
                x1 = vp[a,0,:] + xs[jx] # 1 x 1076
                ix = np.int16(np.round((x1-xmin)/xresolution)) -1
                valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
                                        np.logical_and((ix >=0), (ix < nx)))
                
                cpr[jx,jy,a] = np.sum(im[ix[valid],iy[valid]])
                # valid2 = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
                #                         np.logical_and((ix >=0), (ix < nx)))
                # cpr2[jx,jy,a] = np.sum(im2[ix[valid2],iy[valid2]])

    final = correlation_scaling(cpr)
    
    return final

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def correlation_scaling(x):
    min_val = -1082
    max_val = 1082
    new_min = 0
    new_max = 2
    
    old_range = max_val - min_val
    new_range = new_max - new_min
    scaling_factor = new_range / old_range
    
    output1 = (x - min_val) * scaling_factor + new_min
    return output1

def particle_filter(velocity,encoder_stamps,yaw_rate,imu_stamps,lidar,lidar_stamps,particles):
    [vel,theta,lidar_value,time_value] = time_sync(velocity,encoder_stamps,yaw_rate,imu_stamps,lidar,lidar_stamps)
    velocity_input = np.mean(vel,axis=1)
    # Plot velocity and angles for visualization
    # vel_theta(vel,theta,time_value)

    # Initialsing global map with 0 lidar scan
    origin = meters_to_pixel(0,0)
    global_map = create_empty_map()
    # lidar_value[0:2,:,0] = 0
    # lidar_value[0,0,0] = 0.06

    occupied_cells = lidar_initialise(lidar_value[0:2,:,0],origin)
    log_odds = np.zeros(shape=(global_map['sizex'],global_map['sizey']),dtype=float)
    binary_map = np.zeros(shape=(global_map['sizex'],global_map['sizey']),dtype=float)
    log_odds[:,:] = 0
    binary_map[:,:] = 0
    for i in range(occupied_cells.shape[1]):
        output_my = my_bresenham2D(origin[0],origin[1],occupied_cells[0,i],occupied_cells[1,i])
        log_odds[output_my[0][-1],output_my[1][-1]] = log_odds[output_my[0][-1],output_my[1][-1]] + np.log(4)
        log_odds[output_my[0][:-1],output_my[1][:-1]] = log_odds[output_my[0][:-1],output_my[1][:-1]] - np.log(4)
        binary_map[output_my[0][-1],output_my[1][-1]] = 1
        binary_map[output_my[0][:-1],output_my[1][:-1]] = -1



    position_hypothesis = np.zeros(shape=(4,particles,time_value.shape[0]),dtype=float)
    best = np.zeros(shape=(4,time_value.shape[0]),dtype=float)
    lidar_particles = np.zeros(shape=(particles,3,lidar_value.shape[1]),dtype=float)
    correlation_particles = np.zeros(shape=(particles),dtype=float)
    position_hypothesis[3,:,:] = 1/particles
    fig3 = plt.figure()
    ax1 = fig3.add_subplot(111)

    for i in range(1,10):
        tau = time_value[i] -  time_value[i-1]
        position_hypothesis[0:3,:,i] = motion_model_input_noise(position_hypothesis[0:3,:,i-1],tau,velocity_input[i-1],theta[i-1])
        # move lidar to world frame
        rot_mat = np.zeros(shape=(particles,3,3),dtype=float)
        rot_mat[:,0,0] = np.cos(position_hypothesis[2,:,i])
        rot_mat[:,0,1] = -np.sin(position_hypothesis[2,:,i])
        rot_mat[:,1,0] = np.sin(position_hypothesis[2,:,i])
        rot_mat[:,1,1] = np.cos(position_hypothesis[2,:,i])
        rot_mat[:,2,2] = 1
        trans = np.zeros(shape=(particles,3,1),dtype=float)
        trans[:,0,0] = position_hypothesis[0,:,i]
        trans[:,1,0] = position_hypothesis[1,:,i]
        lidar_particles = rot_mat@lidar_value[:,:,i] + trans
        # lidar_particles[:,0:2,:] = 10
        # lidar_particles[0,0,0] = 0.02
        # lidar_particles[0,1,0] = 0
        # rot_mat = np.zeros(shape=(particles,3,3),dtype=float)
        # rot_mat[:,2,2] = 1
        # rot_mat[:,1,1] = 1
        # rot_mat[:,0,0] = 1
        # lidar_particles = rot_mat@lidar_value[:,:,0]
        
        # finding correlation for each partcile based on observation model
        x_im = np.arange(global_map['xmin'],global_map['xmax']+global_map['res'],global_map['res']) #x-positions of each pixel of the map
        y_im = np.arange(global_map['ymin'],global_map['ymax']+global_map['res'],global_map['res']) #y-positions of each pixel of the map

        left_4units = -4*global_map['res']
        x_range = np.arange(-1*global_map['res'],1*global_map['res'] + global_map['res'],global_map['res'])
        y_range = np.arange(-1*global_map['res'],1*global_map['res'] + global_map['res'],global_map['res'])
        arr_new = lidar_particles.transpose((1, 2, 0))
        correlation = correlation_vector(binary_map,log_odds,x_im,y_im,arr_new[0:2,:,:],x_range,y_range)
        # correlation = my_correlation2(binary_map,log_odds,x_im,y_im,lidar_particles[:,0:2,:],x_range,y_range)
        # if(np.array_equal(correlation,correlation2)):
        #     dummy = 1
        # else:
        #     dummy=2
        
        correlation_particles = np.amax(correlation, axis=(0, 1))
                        
        # # correcting the position error to match correlation
        # position_error_raw = np.argwhere(correlation==correlation_particles)
        # unique, ret_index = np.unique(position_error_raw[:, 2], return_index=True)
        # position_error = position_error_raw[ret_index]
        # position_hypothesis[0,position_error[:,2],i] = position_hypothesis[0,position_error[:,2],i] + x_range[position_error[:,0]-1]
        # position_hypothesis[1,position_error[:,2],i] = position_hypothesis[1,position_error[:,2],i] + y_range[position_error[:,1]]

        # update partcile weights based on correlation
        position_hypothesis[3,:,i] = correlation_particles*position_hypothesis[3,:,i-1]
        position_hypothesis[3,:,i] = position_hypothesis[3,:,i]/np.sum(position_hypothesis[3,:,i])
        alpha_effective = np.sum(np.square(position_hypothesis[3,:,i]))
        N_eff = 1/alpha_effective


        # Map update step
        temp = np.argwhere(position_hypothesis[3,:,i] == np.max(position_hypothesis[3,:,i]))
        most_likely_partcile = temp[0,0]
        best[:,i] = position_hypothesis[:,most_likely_partcile,i]
        origin = meters_to_pixel(position_hypothesis[0,most_likely_partcile,i],position_hypothesis[1,most_likely_partcile,i])
        local_occupied_cells = lidar_initialise(lidar_particles[most_likely_partcile,0:2,:],origin)

        for k in range(local_occupied_cells.shape[1]):
            output_my = my_bresenham2D(origin[0],origin[1],local_occupied_cells[0,k],local_occupied_cells[1,k])
            log_odds[output_my[0][-1],output_my[1][-1]] = log_odds[output_my[0][-1],output_my[1][-1]] + np.log(4)
            log_odds[output_my[0][:-1],output_my[1][:-1]] = log_odds[output_my[0][:-1],output_my[1][:-1]] - np.log(4)
            binary_map[output_my[0][-1],output_my[1][-1]] = 1
            binary_map[output_my[0][:-1],output_my[1][:-1]] = -1

        binary_map = sigmoid(log_odds)
        binary_map = np.where(binary_map > 0.5, 1,np.where(binary_map < 0.5, -1, 0))
        
        N_thresh = particles/5
        if(i%50==0):
            print("partcile resmaple")
            position_hypothesis[:,:,i] = resample(position_hypothesis[:,:,i],particles)

        if(i%100==0 or i==4885):
            
            # occu_grid = sigmoid(log_odds)
            trajectory_pixel = meters_to_pixel(best[0,0:i+1],best[1,0:i+1])
            
            
            # fig4 = plt.figure()
            # plt.imshow(occu_grid,cmap="gray")
            # plt.title("Occupancy Grid Map")
            # name = "/grid/Occupancy2_" + str(i) + ".jpg"
            # filename1 = path + name
            # plt.savefig(filename1,dpi=300, quality=70)

            # output_binary = np.where(log_odds > 0, 1,np.where(log_odds < 0, -1, 0))
            # fig4 = plt.figure()
            # plt.imshow(output_binary,cmap="gray")
            # plt.title("Occupancy Grid Binary Map at T")
            # name5 = "/binary/Binary2_" + str(i) + ".jpg"
            # filename5 = path + name5
            # plt.savefig(filename5,dpi=300, quality=70)

            
            ax1.imshow(binary_map,cmap="gray")
            ax1.plot(trajectory_pixel[1,:],trajectory_pixel[0,:],'r',linewidth=1.5)
            plt.title("Occupancy grid with best trajectory")
            # plt.show()
            name3 = "DS2Final_2" + str(i) + ".jpg"
            filename3 = path + name3
            plt.savefig(filename3,dpi=300, quality=50)


            # plt.close('all')

        if(i%100==0):
            print(i)
        


    return [best,binary_map]

# for abc in range(350,4000,10):
#         origin = meters_to_pixel(0,0)
#         global_map = create_empty_map()
#         occupied_cells = lidar_initialise(lidar_value[0:2,:,abc],origin)
#         log_odds = np.zeros(shape=(global_map['sizex'],global_map['sizey']),dtype=float)
        
#         for i in range(occupied_cells.shape[1]):
#             output_my = my_bresenham2D(origin[0],origin[1],occupied_cells[0,i],occupied_cells[1,i])
#             log_odds[output_my[0][-1],output_my[1][-1]] = log_odds[output_my[0][-1],output_my[1][-1]] + np.log(4)
#             log_odds[output_my[0][:-1],output_my[1][:-1]] = log_odds[output_my[0][:-1],output_my[1][:-1]] - np.log(4)
#             # global_map['map'][output_my[0][-1],output_my[1][-1]] = 1
#             # global_map['map'][output_my[0][:-1],output_my[1][:-1]] = -1
#             # if(i%20==0):
#             #     plt.imshow(global_map['map'],cmap="gray")
#             #     plt.title("map original")

#         global_map['map'] = np.where(log_odds > 0, 1,np.where(log_odds < 0, -1, 0))
        
#         plt.imshow(global_map['map'],cmap="gray")
#         plt.title("Initialized map")