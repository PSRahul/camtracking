#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports
import cv2
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d,art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


#Setting the image paths
cwd = os.getcwd()
image_path=os.path.join(cwd,"task3")
image_list=os.listdir(image_path)
image_list_path=[os.path.join(image_path,x) for x in image_list]


if not os.path.exists('sol_3'):
    os.makedirs('sol_3')


# In[3]:


#Load the database of SIFT points , descriptors and the corresponding 3D coordinates from 1b

import pickle
with open('data/sol_1b.pkl', 'rb') as f:
    b1=pickle.load(f)

kp_3d_all_images=b1[2]
des_all_images=b1[3]
kp_3d_org=[]
des_org=[]
for kp_3d_image in kp_3d_all_images:
    for item in kp_3d_image:
        kp_3d_org.append(np.ravel(item))
        
des_org=[]
for des_image in des_all_images:
    for item in des_image:
        des_org.append(item)  
        
kp_3d_org=np.array(kp_3d_org)
des_org=np.array(des_org)

pnparg = np.load("data/pnparg.npz")
cameraMatrix = pnparg['cameraMatrix']
plotarg=np.load("data/plotarg.npz")
vertices=plotarg['vertices']


# In[4]:


#Find all the sift points of the target image
def target_sift(idx):
    target_img = cv2.imread(image_list_path[idx])
    target_gray = cv.cvtColor(target_img, cv.COLOR_BGR2GRAY)
    target_sift = cv2.xfeatures2d.SIFT_create()
    (target_kp, target_des) = target_sift.detectAndCompute(target_gray, None)
    return target_kp,target_des


# In[5]:


#Runs the brute force matchers and sorts the match by minimum distance.Returns the matched indices of the origin
#image and target image

def matcher(des_org,target_des,num_matches):
    bf = cv2.BFMatcher()
    matches = bf.match(des_org,target_des)
    sorted_matches = sorted(matches, key = lambda x:x.distance)
    if(num_matches==100000):
        num_matches=len(matches)
    selected_matches=sorted_matches[0:num_matches]
    target_index=np.array([item.trainIdx for item in selected_matches])
    origin_index=np.array([item.queryIdx for item in selected_matches])
    unique_idx=np.unique(target_index,return_index=True)[1]
    target_index=target_index[unique_idx]
    origin_index=origin_index[unique_idx]
    num_matches=np.unique(target_index).shape[0]
    return origin_index,target_index,num_matches


# In[6]:


#Get the 3D coordinates from the matched SIFT points in the database

def get_3d_points(origin_index,num_matches):
    objectPoints=np.zeros((num_matches,3)).astype(np.float64)
    for (i,item) in enumerate(origin_index):
            objectPoints[i]=(kp_3d_org[item])
    return objectPoints       


# In[7]:


#Get the 2D points in the target image from the matched SIFT points of the target image
def get_2d_points(target_index,num_matches,target_kp):
    imagePoints=np.zeros((num_matches,2)).astype(np.float64)
    for (i,item) in enumerate(target_index):
            imagePoints[i][0],imagePoints[i][1]=target_kp[item].pt[0],target_kp[item].pt[1]
    return imagePoints        


# In[8]:


#Estimate the pose using RANSAC from the correspondences
#Returns the rotation and the translation vector
def ransac_solver(objectPoints,imagePoints,reprojectionError,iterationsCount):

    _, rvec, tvec,_=cv2.solvePnPRansac(objectPoints ,imagePoints,cameraMatrix,distCoeffs=None,
                                       reprojectionError=reprojectionError,iterationsCount=iterationsCount)
                                       #,flags=cv2.cv2.SOLVEPNP_EPNP)
    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)
    rvec_matrix = cv2.Rodrigues(src=rvec)
    rvec_matrix = np.array(rvec_matrix[0])
    return rvec_matrix, tvec


# In[9]:


#Find vertices in the image for the bounding box
def image_point_finder(rvec_matrix,tvec):
    image_points=np.zeros((8,2))
    for i in range(8):
        world_point=vertices[i].reshape(3,1)
        image_point=np.matmul(cameraMatrix,np.matmul(rvec_matrix,world_point)+tvec)
        image_points[i][0]=image_point[0]/image_point[2]
        image_points[i][1]=image_point[1]/image_point[2]
    return image_points


# In[10]:


#Plot the bounding box from the vertices founds
def pose_plotter(image_points,idx):
    img = cv2.imread(image_list_path[idx])
    implot = plt.imshow(img)
    edges=np.array([2,3,2,6,6,7,3,7,1,0,1,5,4,5,4,0,3,0,7,4,5,6,1,2]).reshape(12,2)
    for e in range(12):
        x=(image_points[edges[e][0]][0],image_points[edges[e][1]][0])
        y=(image_points[edges[e][0]][1],image_points[edges[e][1]][1])
        plt.plot(x,y,'ro-')
        plt.axis('off')
    plt.savefig("sol_3/box/"+image_list[idx],bbox_inches = 'tight',pad_inches = 0)
    plt.close()


# In[11]:


#Function wrapper that calls each of the image on Task2
def ransac_full(num_matches,idx,reprojectionError,iterationsCount):
       
    target_kp,target_des=target_sift(idx)
    origin_index,target_index,num_matches=matcher(des_org,target_des,num_matches)
    objectPoints=get_3d_points(origin_index,num_matches)
    imagePoints=get_2d_points(target_index,num_matches,target_kp)
    rvec_matrix, tvec=ransac_solver(objectPoints,imagePoints,reprojectionError,iterationsCount)
    image_points=image_point_finder(rvec_matrix,tvec)
    pose_plotter(image_points,idx)
    return rvec_matrix,tvec,imagePoints


# In[12]:


#best_num_matches=200
#best_reprojectionError=4
#best_iterationsCount=500
num_matches=200
reprojectionError=4
iterationsCount=500

ransac_index=np.array([0])
for im_number in ransac_index:
    rvec_matrix,tvec,imagePoints=ransac_full(num_matches,im_number,reprojectionError,iterationsCount)
    print("Calculating", "{:.2%}".format(im_number/24))
    load_str="data/sol3_im"+str(im_number)+".npz"
    np.savez(load_str,
        rvec_init=rvec_matrix,
        tvec_init=tvec,
        )


# In[ ]:




