This repository consists of collection of tasks on object detection and pose estimation using OpenCV.

## Task 1

PnP Algorithm for Object Pose Estimation

<code> PnP.py pnp.gif </code>

Finds the pose of the box using the PnP algorithm from the vertices of the box.

![Alt Text](pnp.gif)



## Task 2

Establishing correspondences for SIFT Points 

<code> sift_matching.py sift_matching.gif </code>

Matches the 2D and 3D correspondences of the SIFT points using the Ray-Box intersection method  
![Alt Text](sift_matching.gif)

## Task 3

RANSAC Algorithm for Object Pose Estimation

<code> ransac.py ransac.gif </code>

Uses the RANSAC algorithm to find the object pose from the 2D and 3D correspondences found in the previous task.
![Alt Text](ransac.gif)
## Task 4

Camera Trajectory Estimation by minimising the Reprojection Error 

<code> reprojection_error_initial_pose.py reprojection_error.py reprojection_pose.gif </code>

Finds the pose of the object by minimising the reprojection error of the SIFT points between subsequent object images. 
![Alt Text](reprojection_pose.gif)

Acknowledgements

The tasks and pictures are based on the Tracking and Detection for Computer Vision course offered at TUM.
