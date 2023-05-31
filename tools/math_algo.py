import numpy as np
from numpy import *
import math
from scipy.spatial import ConvexHull
import lap
from pyquaternion import Quaternion



def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = rad2deg(math.atan2(t0, t1))
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = rad2deg(math.asin(t2))
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = rad2deg(math.atan2(t3, t4))-90
    
    return roll_x, pitch_y, yaw_z # in radians

def quaternion_from_euler(rx,ry,rz):
    """ computes the quarternion from euler angles based on the same angle adjustments as euleur_from_quaternion"""
    q = Quaternion(axis=[1, 0, 0], angle=np.deg2rad(rx)) * Quaternion(axis=[0, 1, 0], angle=np.deg2rad(ry)) * Quaternion(axis=[0, 0, 1], angle=np.deg2rad(rz+90))
    return -1*q # not sure why we need the -1 but this works
    
    
def hungarian_algorithm(cost_matrix, thresh):
    # source : https://github.com/ifzhang/ByteTrack/blob/main/tutorials/centertrack/mot_online/matching.py
    # using lap which is a linear assignment problem solver using Jonker-Volgenant algorithm for dense (LAPJV) or sparse (LAPMOD) matrices.
    # the hungarian algorithm is used to solve the linear assignment problem and the LAPJV algorithm is a variant of it.
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b



# 3D IoU caculate code for 3D object detection 
# Kent 2018/12

# source https://github.com/AlienCat-K/3D-IoU-Python/blob/master/3D-IoU-Python.py

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.
   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
   
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])

    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d, inter_vol


def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def GIoU(box1, box2):
    """returns the 3D GIoU (Generalized Intersection over Union) between two boxes
    we do NOT take into account the roatation of the boxes
    source: https://arxiv.org/pdf/1902.09630.pdf"""
    ###### state[-3:] retourne (x, y, z, l, w, h, θ)
    # calculate the IoU in 3D
    box2 = box2["state"]
    x1, y1, z1 = box1[0], box1[1], box1[2]
    x2, y2, z2 = box2[0], box2[1], box2[2]

    l1, w1, h1 = box1[3], box1[4],  box1[5]
    l2, w2, h2 = box2[3], box2[4], box2[5]

    theta1 = box1[6] 
    theta2 = box2[6] 

    corner_3d_box1 = get_3d_box((l1, w1, h1), theta1, (x1, y1, z1))
    corner_3d_box2 = get_3d_box((l2, w2, h2), theta2, (x2, y2, z2))

    IoU3d, _, inter_vol = box3d_iou(corner_3d_box1, corner_3d_box2)

    # calculate the smallest enclosing box volume (cube)
    min_x = min(x1 - l1/2, x2 - l2/2)
    max_x = max(x1 + l1/2, x2 + l2/2)

    min_y = min(y1 - w1/2, y2 - w2/2)
    max_y = max(y1 + w1/2, y2 + w2/2)
    
    min_z = min(z1 - h1/2, z2 - h2/2)
    max_z = max(z1 + h1/2, z2 + h2/2)

    l, w, h = max_x - min_x, max_y - min_y, max_z - min_z
    V_enclosing = l * w * h

    GIoU = IoU3d - ((V_enclosing - (l1 * w1 * h1) - (l2 * w2 * h2) + inter_vol) / V_enclosing)

    return GIoU