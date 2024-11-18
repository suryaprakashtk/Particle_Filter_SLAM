import numpy as np

def motion_model(previous_state,time,vel,ang_vel):
    vx = vel*np.cos(previous_state[2,:])
    vy = vel*np.sin(previous_state[2,:])
    wz = np.ones(shape=(previous_state.shape[1])) * ang_vel
    next_state = previous_state + time*np.array([vx,vy,wz])
    return next_state

def Exact_motion_model(previous_state,time,vel,ang_vel):
    wz = np.ones(shape=(previous_state.shape[1])) * ang_vel

    temp = time*wz*0.5
    temp2 = np.cos(previous_state[2,:] + temp)
    temp3 = np.sin(previous_state[2,:] + temp)
    temp4 = np.sinc(temp)

    vx = vel*temp4*temp2
    vy = vel*temp4*temp3
    
    next_state = previous_state + time*np.array([vx,vy,wz])
    return next_state

def motion_model_input_noise(previous_state,time,vel,ang_vel):
    vx = vel*np.cos(previous_state[2,:]) + np.random.normal(0, 0.01, previous_state.shape[1])
    vy = vel*np.sin(previous_state[2,:]) + np.random.normal(0, 0.01, previous_state.shape[1])
    wz = np.ones(shape=(previous_state.shape[1])) * ang_vel + np.random.normal(0, 0.001, previous_state.shape[1])
    next_state = previous_state + time*np.array([vx,vy,wz])
    return next_state

def motion_model_output_noise(previous_state,time,vel,ang_vel):
    vx = vel*np.cos(previous_state[2,:])
    vy = vel*np.sin(previous_state[2,:])
    wz = np.ones(shape=(previous_state.shape[1])) * ang_vel
    next_state = previous_state + time*np.array([vx,vy,wz])

    x_noise = np.random.normal(0, 0.01, previous_state.shape[1])
    y_noise = np.random.normal(0, 0.01, previous_state.shape[1])
    theta_noise = np.random.normal(0, 0.00001, previous_state.shape[1])
    noise = np.array([x_noise,y_noise,theta_noise])
    next_state = next_state + noise
    return next_state

def Exact_motion_model_input_noise(previous_state,time,vel,ang_vel):
    wz = np.ones(shape=(previous_state.shape[1])) * ang_vel
    wz = wz + np.random.normal(0, 0.001, previous_state.shape[1])
    temp = time*wz*0.5
    temp2 = np.cos(previous_state[2,:] + temp)
    temp3 = np.sin(previous_state[2,:] + temp)
    temp4 = np.sinc(temp)

    vx = vel*temp4*temp2 + np.random.normal(0, 0.05, previous_state.shape[1])
    vy = vel*temp4*temp3 + np.random.normal(0, 0.05, previous_state.shape[1])
    
    next_state = previous_state + time*np.array([vx,vy,wz])
    return next_state

def Exact_motion_model_output_noise(previous_state,time,vel,ang_vel):
    wz = np.ones(shape=(previous_state.shape[1])) * ang_vel
    
    temp = time*wz*0.5
    temp2 = np.cos(previous_state[2,:] + temp)
    temp3 = np.sin(previous_state[2,:] + temp)
    temp4 = np.sinc(temp)

    vx = vel*temp4*temp2
    vy = vel*temp4*temp3
    
    next_state = previous_state + time*np.array([vx,vy,wz])

    x_noise = np.random.normal(0, 0.05, previous_state.shape[1])
    y_noise = np.random.normal(0, 0.05, previous_state.shape[1])
    theta_noise = np.random.normal(0, 0.001, previous_state.shape[1])
    noise = np.array([x_noise,y_noise,theta_noise])
    next_state = next_state + noise
    return next_state



