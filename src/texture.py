import numpy as np
import matplotlib.pyplot as plt
import cv2
from time_sync import *
from map_functions import *

path = "./data/"
disp_path = path + "dataRGBD/Disparity20/"
rgb_path = path + "dataRGBD/RGB20/"

output_path = "./result/texturemap/"

def normalize(img):
    max_ = img.max()
    min_ = img.min()
    return (img - min_)/(max_-min_)

def texture_map_prof(rgbidx,disidx,pose,map):
    imd = cv2.imread(disp_path+'disparity20_' + disidx.astype(str) +'.png',cv2.IMREAD_UNCHANGED) # (480 x 640)
    imc = cv2.imread(rgb_path+'rgb20_' + rgbidx.astype(str) + '.png')[...,::-1] # (480 x 640 x 3)
    disparity = imd.astype(np.float32)
    dd = (-0.00304 * disparity + 3.31)
    z = 1.03 / dd
    v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
    fx = 585.05108211
    fy = 585.05108211
    cx = 315.83800193
    cy = 242.94140713
    x = (u-cx) / fx * z
    y = (v-cy) / fy * z

    x_coords = x.flatten()
    y_coords = y.flatten()
    z_coords = np.ones(shape=(x_coords.shape[0]))
    coords = np.array([x_coords,y_coords,z_coords])
    coords = coords*dd.flatten()
    o_to_r = np.array([[0,-1,0],[0,0,-1],[1,0,0]],dtype=float)
    regular = np.dot(o_to_r.T,coords)

    pitch = 0.36
    yaw = 0.021

    ry = np.zeros(shape=(3,3),dtype=float)
    ry[0,0] = np.cos(pitch)
    ry[0,2] = np.sin(pitch)
    ry[2,0] = -np.sin(pitch)
    ry[2,2] = np.cos(pitch)
    ry[1,1] = 1

    rz = np.zeros(shape=(3,3),dtype=float)
    rz[0,0] = np.cos(yaw)
    rz[0,1] = -np.sin(yaw)
    rz[1,0] = np.sin(yaw)
    rz[1,1] = np.cos(yaw)
    rz[2,2] = 1
    trans = np.zeros(shape=(3,1),dtype=float)
    trans[0,0] = 0.18
    trans[1,0] = 0.005
    trans[2,0] = 0.36

    temp = np.dot(rz,ry)
    camera_frame = np.dot(temp,regular) + trans

    rot_mat = np.zeros(shape=(3,3),dtype=float)
    rot_mat[0,0] = np.cos(pose[2])
    rot_mat[0,1] = -np.sin(pose[2])
    rot_mat[1,0] = np.sin(pose[2])
    rot_mat[1,1] = np.cos(pose[2])
    rot_mat[2,2] = 1
    translation = np.zeros(shape=(3,1),dtype=float)
    translation[0,0] = pose[0]
    translation[1,0] = pose[1]
    translation[2,0] = 0
    r_world = np.dot(rot_mat,camera_frame) + translation
    valid_index = np.logical_and((r_world[2,:] >=0), (r_world[2,:] <=0.04))
    origin = meters_to_pixel(0,0)
    world_pixel_coords = texture_cells(r_world[0:2,:],origin)


    rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
    rgbv = np.round((v * 526.37 + 16662.0)/fy)
    valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])
    rgbu_flat = rgbu.flatten()
    rgbv_flat = rgbv.flatten()
    rgbu_flat_int = rgbu_flat.astype(int)
    rgbv_flat_int = rgbv_flat.astype(int)
    valid = (rgbu_flat>= 0)&(rgbu_flat < disparity.shape[1])&(rgbv_flat>=0)&(rgbv_flat<disparity.shape[0])
    final_valid_index = np.logical_and((valid ==True), (valid_index == True))
    map[world_pixel_coords[0,final_valid_index],world_pixel_coords[1,final_valid_index],:] = imc[rgbv_flat_int[final_valid_index],rgbu_flat_int[final_valid_index],:]

    return map
    # fig = plt.figure(figsize=(10, 13.3))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(z[valid],-x[valid],-y[valid],c=imc[rgbv[valid].astype(int),rgbu[valid].astype(int)]/255.0)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.view_init(elev=0, azim=180)
    # plt.show()

    # # display disparity image
    # plt.imshow(normalize(imd), cmap='gray')
    # plt.show()

# texture_map()
# print("halt")


def texture_map(encodertime,rgbtime,disptime,trajectory,time):
    timemap = time_sync_rgb(time,rgbtime,disptime)
    global_map = create_empty_map()
    texture_map = np.zeros(shape=(global_map['sizex'],global_map['sizey'],3),dtype=int)
    for i in range(2,timemap.shape[0]):

        texture_map = texture_map_prof(timemap[i,1],timemap[i,0],trajectory[0:3,timemap[i,2]],texture_map)
        print(i)
    
    fig4 = plt.figure()
    plt.imshow(texture_map)
    plt.title("Texture Map for Data set 2")
    plt.show
    name4 = "TextureMapDs2" + ".jpg"
    filename4 = output_path + name4
    plt.savefig(filename4,dpi=300, quality=50)

    return texture_map