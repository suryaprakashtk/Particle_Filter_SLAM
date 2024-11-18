import numpy as np

def time_sync(velocity,encoder_stamps,yaw_rate,imu_stamps,lidar,lidar_stamps):
    # timesync for all three sensor datas
    no_of_imudata = yaw_rate.shape[0]
    no_of_encoderdata = velocity.shape[0]
    no_of_lidar = lidar.shape[2]
    timesteps_map = np.zeros(shape=(max(no_of_imudata,no_of_encoderdata,no_of_lidar),3),dtype=int)
    first_common_time = max(imu_stamps[0],encoder_stamps[0])
    # syncing imu and encoder
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
    first_common_time = max(lidar_stamps[0],encoder_stamps[0])
    # syncing imu and encoder
    no_of_sync = k
    k=0
    j=0
    for i in range(no_of_sync):
        if(encoder_stamps[timemap[i,0]]<first_common_time):
            continue
        while(j<no_of_lidar):
            if(encoder_stamps[timemap[i,0]]<lidar_stamps[j]):
                timemap[k,2] = j-1
                k=k+1
                break
            j=j+1
    yaw_rate_time = np.zeros(shape = (timemap.shape[0]),dtype=float)
    time_series = np.zeros(shape=(timemap.shape[0] + 1),dtype=float)
    for i in range(timemap.shape[0]):
        if(i==0):
            tau = encoder_stamps[timemap[i,0]] - encoder_stamps[timemap[i,0]-1]
            yaw_rate_time[i]  = np.mean(yaw_rate[timemap[i,1]-1:timemap[i,1]])
        else:
            tau = encoder_stamps[timemap[i,0]] - encoder_stamps[timemap[i-1,0]]
            # average value of wz from t-1 to t instance
            # at i=70 the timestamps of IMU got mapped to same
            if(timemap[i-1,1]!=timemap[i,1] and i<timemap.shape[0]):
                yaw_rate_time[i]  = np.mean(yaw_rate[timemap[i-1,1]:timemap[i,1]+1])
            else:
                yaw_rate_time[i]  = yaw_rate[timemap[i,1]]
        time_series[i+1] = time_series[i] + tau

    velocity_time = velocity[timemap[:,0],:]
    lidar_time = lidar[:,:,timemap[:,2]]

    return [velocity_time,yaw_rate_time,lidar_time,time_series]

def time_sync_rgb(encodertime,rgbtime,disptime):
    # timesync for all three sensor datas
    no_of_rgb = rgbtime.shape[0]
    no_of_encoderdata = encodertime.shape[0]
    no_of_disp = disptime.shape[0]
    timesteps_map = np.zeros(shape=(max(no_of_rgb,no_of_encoderdata,no_of_disp),3),dtype=int)
    first_common_time = max(rgbtime[0],disptime[0],encodertime[0])
    # syncing imu and encoder
    k=0
    j=0
    for i in range(no_of_disp):
        if(disptime[i]<first_common_time):
            continue
        while(j<no_of_rgb):
            if(disptime[i]<rgbtime[j]):
                timesteps_map[k,0] = i
                timesteps_map[k,1] = j-1
                k=k+1
                break
            j=j+1
    # 0 index is encoder velocity and 1 index is IMU data yaw rate
    timemap = timesteps_map[:k]

    # syncing imu and encoder
    no_of_sync = k
    k=0
    j=0
    for i in range(no_of_sync):
        if(disptime[timemap[i,0]]<first_common_time):
            continue
        while(j<no_of_encoderdata):
            if(disptime[timemap[i,0]]<encodertime[j]):
                timemap[k,2] = j-1
                k=k+1
                break
            j=j+1

    return timemap