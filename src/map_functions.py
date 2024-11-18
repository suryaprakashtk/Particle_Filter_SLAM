import numpy as np
import matplotlib.pyplot as plt; plt.ion()


def my_bresenham2D(sx, sy, ex, ey):
    '''
    Bresenham's ray tracing algorithm in 2D.
    Inputs:
      (sx, sy)	start point of ray
      (ex, ey)	end point of ray
    '''
    sx = int(sx)
    sy = int(sy)
    ex = int(ex)
    ey = int(ey)
    dx = abs(ex-sx)
    dy = abs(ey-sy)
    steep = abs(dy)>abs(dx)
    if steep:
      dx,dy = dy,dx # swap 

    if dy == 0:
      q = np.zeros((dx+1,1))
    else:
      q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
    if steep:
      if sy <= ey:
        y = np.arange(sy,ey+1)
      else:
        y = np.arange(sy,ey-1,-1)
      if sx <= ex:
        x = sx + np.cumsum(q)
      else:
        x = sx - np.cumsum(q)
    else:
      if sx <= ex:
        x = np.arange(sx,ex+1)
      else:
        x = np.arange(sx,ex-1,-1)
      if sy <= ey:
        y = sy + np.cumsum(q)
      else:
        y = sy - np.cumsum(q)
    return np.array([x,y],dtype=int)

def my_show_lidar(ranges):
    # plots 1 lidar scan data in a circle
    # copied from pr2_utils file
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    plt.figure()
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, ranges)
    ax.set_rmax(10)
    ax.set_rticks([0.5, 1, 1.5, 2])  # fewer radial ticks
    ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.grid(True)
    ax.set_title("Lidar scan data", va='bottom')
    plt.show()
    print("my_show_lidar executed")

def create_empty_map():
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -40  #meters
    MAP['ymin']  = -40
    MAP['xmax']  =  40
    MAP['ymax']  =  40 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8
    return MAP

def meters_to_pixel(x_meters,y_meters):
    map = create_empty_map()
    x_pixel = np.ceil((x_meters - map['xmin']) / map['res'] ).astype(np.int16)-1
    y_pixel = np.ceil((y_meters - map['ymin']) / map['res'] ).astype(np.int16)-1
    occupied_pixels = np.array([x_pixel,y_pixel])
    return occupied_pixels


def lidar_initialise(coordinates,origin):
    map = create_empty_map()
    xis = np.ceil((coordinates[0,:] - map['xmin']) / map['res'] ).astype(np.int16)-1
    yis = np.ceil((coordinates[1,:] - map['ymin']) / map['res'] ).astype(np.int16)-1
    indGood = np.logical_and(np.logical_and(np.logical_and((xis >= 0), (yis >= 0)), (xis < map['sizex'])), (yis < map['sizey']))
    ind = np.logical_not(np.logical_and(xis == origin[0],yis == origin[1]))
    ind_final = np.logical_and(indGood==True,ind==True)
    occupied_pixels = np.array([xis[ind_final],yis[ind_final]])
    return occupied_pixels

def texture_cells(coordinates,origin):
    map = create_empty_map()
    xis = np.ceil((coordinates[0,:] - map['xmin']) / map['res'] ).astype(np.int16)-1
    yis = np.ceil((coordinates[1,:] - map['ymin']) / map['res'] ).astype(np.int16)-1
    occupied_pixels = np.array([xis,yis])
    return occupied_pixels

def lidar_to_pixel(coordinates,origin):
    map = create_empty_map()
    xis = np.ceil((coordinates[0,:] - map['xmin']) / map['res'] ).astype(np.int16)-1
    yis = np.ceil((coordinates[1,:] - map['ymin']) / map['res'] ).astype(np.int16)-1
    indGood = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and((xis >= 0), (yis >= 0)), (xis < map['sizex'])), (yis < map['sizey'])), (xis != origin[0])), (yis != origin[1]))
    occupied_pixels = np.array([xis[indGood],yis[indGood]])
    return occupied_pixels


# origin = np.array([0,0])
# occupied_cells = np.array([[5,2],[7,3]])
# log_odds = np.zeros(shape=(10,10),dtype=float)
# for i in range(occupied_cells.shape[1]):
#     output_my = my_bresenham2D(origin[0],origin[1],occupied_cells[0,i],occupied_cells[1,i])
#     log_odds[output_my[0][-1],output_my[1][-1]] = log_odds[output_my[0][-1],output_my[1][-1]] + np.log(4)
#     log_odds[output_my[0][:-1],output_my[1][:-1]] = log_odds[output_my[0][:-1],output_my[1][:-1]] - np.log(4)
#     print("sdf")
# map = create_empty_map()