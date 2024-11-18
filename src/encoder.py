import numpy as np
from scipy.linalg import expm, sinm, cosm

def encode_data_manipulation(encoder_counts,encoder_time):
    distance_count = np.cumsum(encoder_counts,axis=1)
    #  wheel travels 0.0022 m per tic
    distance_m = distance_count*0.0022
    distance_right_m = np.mean(distance_m[(0,2),:],axis = 0)
    distance_left_m = np.mean(distance_m[(1,3),:],axis = 0)
    # right and left wheel cumulative distance in meters
    distance_rl_m = np.column_stack((distance_right_m,distance_left_m))
    velcity = np.zeros(shape = (encoder_counts.shape[1]-1,4),dtype=float)
    for i in range(velcity.shape[0]):
        t_diff = encoder_time[i+1] - encoder_time[i]
        velcity[i,0] = (distance_m[0,i+1] - distance_m[0,i])/t_diff
        velcity[i,1] = (distance_m[1,i+1] - distance_m[1,i])/t_diff
        velcity[i,2] = (distance_m[2,i+1] - distance_m[2,i])/t_diff
        velcity[i,3] = (distance_m[3,i+1] - distance_m[3,i])/t_diff
    
    velocity_right = np.mean(velcity[:,(0,2)],axis = 1)
    velocity_left = np.mean(velcity[:,(1,3)],axis = 1)
    # right and left wheel velocity in meters/sec
    velocity_rl = np.column_stack((velocity_right,velocity_left))
    print("encode_data_manipulation executed")
    return [distance_rl_m,velocity_rl]

def motion_model(velocity,yaw_rate,imu_stamps,encoder_stamps):
    # matching the timestamps of encoder to imu
    # for all encoder a in the past imu data is mapped
    no_of_imudata = yaw_rate.shape[0]
    no_of_encoderdata = velocity.shape[0]
    timesteps_map = np.zeros(shape=(max(no_of_imudata,no_of_encoderdata),2),dtype=int)
    first_common_time = max(imu_stamps[0],encoder_stamps[0])
    k=0
    j=0
    for i in range(no_of_encoderdata):
        if(encoder_stamps[i]<first_common_time):
            continue
        while(j<no_of_imudata):
            if(encoder_stamps[i]<imu_stamps[j]):
                timesteps_map[k,0] = i
                timesteps_map[k,1] = j-1
                k=k+1
                break
            j=j+1
    # 0 index is encoder velocity and 1 index is IMU data yaw rate
    timemap = timesteps_map[:k]

    # Pose estimation over time
    pose_trajectory = np.zeros(shape=(4,4,timemap.shape[0] + 1),dtype=float)
    time_series = np.zeros(shape=(timemap.shape[0] + 1),dtype=float)
    # homogenous coordiantes
    pose_trajectory[3,3,:] = 1
    # initial pose is identity
    pose_trajectory[:,:,0] = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=float)
    for i in range(0,pose_trajectory.shape[2]-1):
        # twist in body frame
        wx = 0.0
        wy = 0.0
        wz = yaw_rate[timemap[i,1]]
        vx = (velocity[timemap[i,0],0] + velocity[timemap[i,0],1])*0.5
        vy = 0.0
        vz = 0.0
        if(i==0):
            tau = encoder_stamps[timemap[i,0]] - encoder_stamps[timemap[i,0]-1]
        else:
            tau = encoder_stamps[timemap[i,0]] - encoder_stamps[timemap[i-1,0]]
        twist_hat = np.array([[0,-wz,wy,vx],[wz,0,-wx,vy],[-wy,wx,0,vz],[0,0,0,0]])
        twist_hat = tau * twist_hat
        twist_exp = expm(twist_hat)
        pose_trajectory[:,:,i+1] = np.matmul(pose_trajectory[:,:,i],twist_exp)
        time_series[i+1] = time_series[i] + tau
    print("motion_model executed")
    return [pose_trajectory,time_series]

def dead_recokon(velocity,yaw_rate,imu_stamps,encoder_stamps):
    # matching the timestamps of encoder to imu
    # for all encoder a in the past imu data is mapped
    no_of_imudata = yaw_rate.shape[0]
    no_of_encoderdata = velocity.shape[0]
    timesteps_map = np.zeros(shape=(max(no_of_imudata,no_of_encoderdata),2),dtype=int)
    first_common_time = max(imu_stamps[0],encoder_stamps[0])
    k=0
    j=0
    for i in range(no_of_encoderdata):
        if(encoder_stamps[i]<first_common_time):
            continue
        while(j<no_of_imudata):
            if(encoder_stamps[i]<imu_stamps[j]):
                timesteps_map[k,0] = i
                timesteps_map[k,1] = j-1
                k=k+1
                break
            j=j+1
    # 0 index is encoder velocity and 1 index is IMU data yaw rate
    timemap = timesteps_map[:k]

    # Pose estimation over time
    xy_trajectory = np.zeros(shape=(3,timemap.shape[0] + 1),dtype=float)
    time_series = np.zeros(shape=(timemap.shape[0] + 1),dtype=float)
    # initial pose is identity
    xy_trajectory[:,0] = np.array([0,0,0],dtype=float)
    for i in range(0,xy_trajectory.shape[1]-1):
        if(i==0):
            tau = encoder_stamps[timemap[i,0]] - encoder_stamps[timemap[i,0]-1]
            wz = np.mean(yaw_rate[timemap[i,1]-1:timemap[i,1]])
        else:
            tau = encoder_stamps[timemap[i,0]] - encoder_stamps[timemap[i-1,0]]
            # average value of wz from t-1 to t instance
            # at i=70 the timestamps of IMU got mapped to same
            if(timemap[i-1,1]!=timemap[i,1]):
                wz = np.mean(yaw_rate[timemap[i-1,1]:timemap[i,1]])
            else:
                wz = yaw_rate[timemap[i,1]]

        vx = (velocity[timemap[i,0],0] + velocity[timemap[i,0],1])*0.5*np.cos(xy_trajectory[2,i])
        vy = (velocity[timemap[i,0],0] + velocity[timemap[i,0],1])*0.5*np.sin(xy_trajectory[2,i])

        xy_trajectory[:,i+1] = xy_trajectory[:,i] + tau*np.array([vx,vy,wz])
        time_series[i+1] = time_series[i] + tau
    print("motion_model_xytheta executed")
    return [xy_trajectory,time_series]