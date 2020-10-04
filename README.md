# Extended Kalman Filter 

This repository contains code for EKF implementation in python for an open source Lidar data. 

The Goal of a Kalman Filter is to take a Probabilistic Estimate of the state and update it in real time in two steps, Prediction and Correction Step.

​                                                  <img src="EKF/images/KF.png"  width="800"/>
​     

#### A. Linearizing a Non Linear Function
In 2D, Choose an operating point 'a' approximate the non-linear function by a tangent line at the point. 



<img src="EKF/images/EKF.png" width="800"/>



For EKF we choose the operating point to be our most recent state estimate, our known input and zero noise. 


We will be computing the Jacobians of the Non-Linear Measurement and Motion Model w.r.t the posterior state and the noise.

In vector calculus jacobian matrix is the matrix of all first order partial derivatives of a vector values function. 

Intuitively the Jacobian matrix tells you how fast each output of the function is changing along each input direction.



#### Further Information about EKF can be found in the file (Nonlinear Kalman Filter - Extended Kalman Filter.pdf)



## Data :

##### The data contains lidar and control inputs information (linear and angular velocity).



## Setup : 

### Step 1 : Setting up Perquisites 

```
pip install numpy 
pip install matplotlib
pip install pickle
pip install matplotlib
pip install easydict
```

### Step 2 : Setting up parameters in the yaml file.

Setup the variance, data file path and other parameters in the yaml file. 

### Step 3 : Running main.py 

```
python main.py 
```



## Results : 

<img src="EKF/images/Trajectory.png" width="500"/>     <img src="EKF/images/Theta.png" width="500"/>





