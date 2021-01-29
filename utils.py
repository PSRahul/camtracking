#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


#Image points and Object Points for Task1

cameraMatrix = np.array([2960.37845, 0, 1841.68855,
                         0, 2960.37845, 1235.23369,
                         0, 0, 1
                         ]).reshape(3, 3)

objectPoints_0 = np.array([0, 0.063, 0.093,
                           0.165, 0.063, 0.093,
                           0.165, 0, 0.093,
                           0, 0, 0.093,
                           0.165, 0, 0,
                           0, 0, 0
                           ]).reshape(6, 3).astype(np.float32)

objectPoints_1 = np.array([0, 0.063, 0.093,
                           0.165, 0.063, 0.093,
                           0.165, 0, 0.093,
                           0, 0, 0.093,
                           0.165, 0.063, 0,
                           0.165, 0, 0,
                           0, 0, 0
                           ]).reshape(7, 3).astype(np.float32)

objectPoints_2 = np.array([0, 0.063, 0.093,
                           0.165, 0.063, 0.093,
                           0.165, 0, 0.093,
                           0, 0, 0.093,
                           0.165, 0.063, 0,
                           0.165, 0, 0

                           ]).reshape(6, 3).astype(np.float32)

objectPoints_3 = np.array([0, 0.063, 0.093,
                           0.165, 0.063, 0.093,
                           0.165, 0, 0.093,
                           0, 0, 0.093,
                           0, 0.063, 0,
                           0.165, 0.063, 0,
                           0.165, 0, 0

                           ]).reshape(7, 3).astype(np.float32)

objectPoints_4 = np.array([0, 0.063, 0.093,
                           0.165, 0.063, 0.093,
                           0.165, 0, 0.093,
                           0, 0, 0.093,
                           0, 0.063, 0,
                           0.165, 0.063, 0

                           ]).reshape(6, 3).astype(np.float32)

objectPoints_5 = np.array([0, 0.063, 0.093,
                           0.165, 0.063, 0.093,
                           0.165, 0, 0.093,
                           0, 0, 0.093,
                           0, 0.063, 0,
                           0.165, 0.063, 0,
                           0, 0, 0
                           ]).reshape(7, 3).astype(np.float32)

objectPoints_6 = np.array([0, 0.063, 0.093,
                           0.165, 0.063, 0.093,
                           0.165, 0, 0.093,
                           0, 0, 0.093,
                           0, 0.063, 0,
                           0, 0, 0
                           ]).reshape(6, 3).astype(np.float32)

objectPoints_7 = np.array([0, 0.063, 0.093,
                           0.165, 0.063, 0.093,
                           0.165, 0, 0.093,
                           0, 0, 0.093,
                           0, 0.063, 0,
                           0.165, 0, 0,
                           0, 0, 0
                           ]).reshape(7, 3).astype(np.float32)

imagePoints_0_tlc = np.array([1373, 1019,
                              2238, 1006,
                              2312, 1116,
                              1342, 1139,
                              2280, 1590,
                              1374, 1617
                              ]).reshape(6, 2).astype(np.float32)

imagePoints_1_tlc = np.array([1659, 932,
                              2202, 1152,
                              1926, 1233,
                              1401, 986,
                              2177, 1617,
                              1913, 1731,
                              1418, 1406
                              ]).reshape(7, 2).astype(np.float32)

imagePoints_2_tlc = np.array([1986, 855,
                              1938, 1147,
                              1536, 1146,
                              1584, 854,
                              1922, 1649,
                              1548, 1657]).reshape(6, 2).astype(np.float32)

imagePoints_3_tlc = np.array([2315, 975,
                              1706, 1194,
                              1460, 1098,
                              2076, 907,
                              2287, 1393,
                              1713, 1687,
                              1476, 1565
                              ]).reshape(7, 2).astype(np.float32)

imagePoints_4_tlc = np.array([2294, 1125,
                              1305, 1119,
                              1354, 995,
                              2242, 1004,
                              2261, 1604,
                              1337, 1598
                              ]).reshape(6, 2).astype(np.float32)

imagePoints_5_tlc = np.array([1744, 1176,
                              1317, 934,
                              1584, 882,
                              2054, 1101,
                              1758, 1667,
                              1348, 1347,
                              2045, 1573
                              ]).reshape(7, 2).astype(np.float32)

imagePoints_6_tlc = np.array([1598, 1186,
                          1645, 906,
                          1939, 905,
                          1983, 1187,
                          1610, 1679,
                          1972, 1678
                          ]).reshape(6, 2).astype(np.float32)

imagePoints_7_tlc = np.array([1455, 1142,
                              2054, 965,
                              2297, 1027,
                              1702, 1233,
                              1466, 1610,
                              2268, 1448,
                              1700, 1730
                              ]).reshape(7, 2).astype(np.float32)


image_plane_center = np.array([1818, 1128,
                               1928, 1238,
                               1743, 1143,
                               1707, 1197,
                               1794, 1122,
                               1743, 1179,
                               1788, 1188,
                               1700, 1234
                               ]).reshape(8, 2).astype(np.float32)

image_plane_center_zero = np.array([0,0,
                               0,0,
                               0,0,
                               0,0,
                               0,0,
                               0,0,
                               0,0,
                               0,0,
                               ]).reshape(8, 2).astype(np.float32)

imagePoints_0 = imagePoints_0_tlc-image_plane_center_zero[0]
imagePoints_1 = imagePoints_1_tlc-image_plane_center_zero[1]
imagePoints_2 = imagePoints_2_tlc-image_plane_center_zero[2]
imagePoints_3 = imagePoints_3_tlc-image_plane_center_zero[3]
imagePoints_4 = imagePoints_4_tlc-image_plane_center_zero[4]
imagePoints_5 = imagePoints_5_tlc-image_plane_center_zero[5]
imagePoints_6 = imagePoints_6_tlc-image_plane_center_zero[6]
imagePoints_7 = imagePoints_7_tlc-image_plane_center_zero[7]


# In[3]:


import os
if not os.path.exists('data'):
    os.makedirs('data')

np.savez('data/pnparg.npz', cameraMatrix=cameraMatrix,
         objectPoints_0=objectPoints_0,
         objectPoints_1=objectPoints_1,
         objectPoints_2=objectPoints_2,
         objectPoints_3=objectPoints_3,
         objectPoints_4=objectPoints_4,
         objectPoints_5=objectPoints_5,
         objectPoints_6=objectPoints_6,
         objectPoints_7=objectPoints_7,
         imagePoints_0_tlc=imagePoints_0_tlc,
         imagePoints_1_tlc=imagePoints_1_tlc,
         imagePoints_2_tlc=imagePoints_2_tlc,
         imagePoints_3_tlc=imagePoints_3_tlc,
         imagePoints_4_tlc=imagePoints_4_tlc,
         imagePoints_5_tlc=imagePoints_5_tlc,
         imagePoints_6_tlc=imagePoints_6_tlc,
         imagePoints_7_tlc=imagePoints_7_tlc,
         image_plane_center=image_plane_center,
         image_plane_center_zero=image_plane_center_zero,
         imagePoints_0=imagePoints_0,
         imagePoints_1=imagePoints_1,
         imagePoints_2=imagePoints_2,
         imagePoints_3=imagePoints_3,
         imagePoints_4=imagePoints_4,
         imagePoints_5=imagePoints_5,
         imagePoints_6=imagePoints_6,
         imagePoints_7=imagePoints_7)


# In[4]:


#Points for generating the 3D plot

poly3d = [[(0.0, 0.063, 0.093),
           (0.165, 0.063, 0.093),
           (0.165, 0.063, 0.0),
           (0.0, 0.063, 0.0)],
          [(0.0, 0.0, 0.093), (0.165, 0.0, 0.093),
           (0.165, 0.0, 0.0), (0.0, 0.0, 0.0)],
          [(0.0, 0.063, 0.093),
           (0.165, 0.063, 0.093),
           (0.0, 0.0, 0.093),
           (0.165, 0.0, 0.093)],
          [(0.0, 0.063, 0.0), (0.165, 0.063, 0.0),
           (0.165, 0.0, 0.0), (0.0, 0.0, 0.0)],
          [(0.165, 0.063, 0.093),
           (0.165, 0.0, 0.093),
           (0.165, 0.063, 0.0),
           (0.165, 0.0, 0.0)],
          [(0.0, 0.063, 0.093), (0.0, 0.063, 0.0), (0.0, 0.0, 0.093), (0.0, 0.0, 0.0)],
          [(0.0, 0.063, 0.093), (0.0,0.0, 0.093), (0.0, 0.063,0.0), (0.0, 0.0, 0.0)]]

vertices = np.array([
    0, 0.063, 0.093,
    0.165, 0.063, 0.093,
    0.165, 0, 0.093,
    0, 0, 0.093,
    0, 0.063, 0,
    0.165, 0.063, 0,
    0.165, 0, 0,
    0, 0, 0
]).reshape(8, 3).astype(np.float32)

world_plane_center = np.array([0.0825, 0, 0.093,
                               0.165, 0, 0.093,
                               0.165, 0.0315, 0.093,
                               0.165, 0.063, 0.093,
                               0.0825, 0.063, 0.093,
                               0, 0.063, 0.093,
                               0, 0.0315, 0.093,
                               0, 0, 0.093

                               ]).reshape(8, 3).astype(np.float32)



# In[5]:


np.savez('data/plotarg.npz', poly3d=poly3d,
         vertices=vertices,
         world_plane_center=world_plane_center)


# In[ ]:




