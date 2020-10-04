"""
Author - Kailash Nagarajan
Date - 3-Oct-2020
email-ID - kailashnagarajan@gmail.com 
"""

import pickle 
import numpy as np 
import matplotlib.pyplot as plt 
import EKF_Estimation
from parameters.parameters import read_params

"""
Main Script - Handles the jacobians, Motion,Measurement Models,reading the data file and helper functions. 
"""

params = read_params() # Loads Parameters from the yaml file.

def read_data_from_pickle():

    """
    This function reads the data from the pickle file and returns the different data points.
    
    Returns :
    controls -> List containing Control data (Linear Velocity and Angular Velocity) - List - (2,)

    lidar_data -> List containing Lidar data (bearing,range,landmark position,
                 distance b/w sensor and robot COG) - List - (4,)

    variance -> List of Variance values (linear and angular velocity variance,
                bearing variance,range variance) - List - (4,)

    initial_values -> List Containing Initial State Values (x,y,theta)
    
    t -> Time array - Array - (,501)

    """

    #Reading data from the pickle File
    with open(params.filename, 'rb') as f : 
        data = pickle.load(f)

    #Assigning Data from the Pickle file.

    t = data['t'] #time 

    #initial values
    x_init = data['x_init']   #initial value for x 
    y_init = data['y_init']   #initial value for y 
    th_init = data['th_init'] #initial value for heading

    #variance 
    v_var = data['v_var'] #linear velocity varaiance
    om_var = data['om_var'] #angular velocity variance
    b_var = data['b_var'] #LIDAR bearing variance
    r_var = data['r_var'] #LIDAR range variance

    #control inputs
    v = data['v'] #linear velocity
    om = data['om'] #angular velocity

    #sensor measurements
    b = data['b'] #bearing readings of the lidar.
    l = data['l'] #global position of the landmarks.
    r = data['r'] #range readings of the lidar.
    d = data['d'] #distance between robot center and lidar origin.

    controls = [v,om]
    lidar_data = [b,l,r,d]
    variance = [v_var,om_var,b_var,r_var]
    initial_values = [x_init,y_init,th_init]


    return controls,lidar_data,variance,initial_values,t


def parameters_intialization():

    """
    This function intializes all the required parameters for EKF. 
    
    Returns: 
    Q -> Process Noise Co-Variance Matrix - Array - (2,2)
    R -> Measurement Noise Co-Variance Matrix - Array - (2,2)
    x_est -> Initial State Estimate Matrix - Array - (501,3)
    P_est -> Initial Co-variance Estimate Matrix - Array - (501,3,3) 

    """

    controls,lidar_data,variance,initial_values,t = read_data_from_pickle()

    v_var = params.v_var # Linear Velocity Variance
    om_var = params.om_var # Angular Velocity Variance
    r_var = params.r_var # Lidar Range Variance
    b_var = params.b_var # Lidar bearing Variance

    # v_var = variance[0]
    # om_var = variance[1]
    # r_var = variance[2]
    # b_var = variance[3]

    Q = np.diag([v_var,om_var]) # Intializing Process Noise Co-Variance Matrix
    R = np.diag([r_var,b_var]) # Intializing Process Noise Co-Variance Matrix

    x_est = np.zeros([len(controls[0]),3]) # Initializing State Estimate Matrix
    P_est = np.zeros([len(controls[0]),3,3]) # Initializing Co-Variance Matrix

    x_est[0] = np.array([initial_values[0],initial_values[1],initial_values[2]]) # Initial State at t = 0
    P_est[0] = np.diag([1,1,0.1]) # Initial Co-Variance at t = 0

    return Q,R,x_est,P_est


def measurement_model(x_l,y_l,x_current,y_current,theta_current,d):

    """
    The Measurement Model Function takes the states as input and gives out the 'y'
    
    Parameters :
    x_l -> Global Lidar Landmark Positon in X Direction - float val
    y_l -> Gloabal Lidar Landmark Position in Y Direction - float val
    x_current -> Current X-Position of the Robot - float val 
    y_current -> Current Y-Position of the Robot - float val 
    theta_current -> Current theta of the Robot - float val 
    d -> Distance between Lidar center and Robot COG - float val

    Returns: 

    y_k - Measurement Output - float val

    """

    y_k = np.array([[np.sqrt((x_l-x_current-d*np.cos(theta_current))^2 + (y_l-y_current-d*np.sin(theta_current))^2)], \
          [np.arctan2(y_k-y_current-d*np.sin(theta_current),x_l-x_current-d*np.cos(theta_current))-theta_current]])

    return y_k 

def motion_model(v,om,x,y,th,delta_t):

    """
    The Motion Model function takes the current state and outputs the next state 
    
    Parameters:

    v -> Linear Velocity - float val.
    om -> Angular Velocity - float val. 
    x -> Current x-position of the robot - float val 
    y -> Current y-position of the robot - float val 
    th -> Current theta of the robot - float val 
    delta_t -> Time Period - float val 

    Return:

    new_state -> Next states (x,y,theta) - Array - (3,)

    """
    
    state_matrix = np.array([[np.cos(th),0],[np.sin(th),0],[0,1]], dtype='float') # State Matrix
    control_input = np.array([[v],[om]]) # Control Input Matrix 
    prev_state = np.array([[x],[y],[th]]) # Previous State Matrix
    new_state = prev_state + state_matrix.dot(control_input).dot(delta_t) # Next State matrix
    return new_state[0][0],new_state[1][0],new_state[2][0]


def jacobian_measurement_model(x_l,x_k,y_l,y_k,d,theta_k):

    """
    The Jacobian of the Measurement Model w.r.t Noise and last state is calculated using this function

    Parameters : 
    x_l -> The x-position of lidar global landmark - float val
    y_l -> The y-position of lidar global landmark - float val
    x_k -> Current Position of the Robot - float val
    y_k -> Current Position of the Robot - float val
    theta_k -> Current Angular Position of the Robot - float val
    d -> Distanc between the lidar center and Robot COG - float val

    Returns : 
    H -> Measurement Jacobian w.r.t posterior - float val
    M -> Measurement Jacobian w.r.t Noise - float val 
    r -> Range - float val
    phi -> Bearing - float val

    """

    d_x = x_l - x_k - d*np.cos(theta_k) # Distance Between Landmark and Robot COG in x-direction
    d_y = y_l - y_k - d*np.sin(theta_k) # Distance Between Landmark and Robot COG in y-direction

    r = np.sqrt(d_x**2+d_y**2) # Range of the Lidar 
    phi = np.arctan2(d_y,d_x)-theta_k # Bearing of the Lidar 
    
    """
    Currently the Jacobian is calculated on MATLAB and copied here, 
    TODO : Move Calculation of Jacobian to Python

    """

    H = np.zeros((2,3))
    H[0,0] = -d_x/r
    H[0,1] = -d_y/r
    H[0,2] = d*(d_x*np.sin(theta_k) - d_y*np.cos(theta_k))/r
    H[1,0] = d_y/r**2
    H[1,1] = -d_x/r**2
    H[1,2] = -1-d*(d_y*np.sin(theta_k) + d_x*np.cos(theta_k))/r**2

    M = np.identity(2)

    return H ,M ,r,phi

def jacobian_motion_model(v_k_1,theta,delta_t):

    """
    The Jacobian of the Motion Model w.r.t Noise and Last State has been calculated using this function
    
    Parameters : 
    v_k_1 - Previous Linear Velocity - float val 
    delta_t - Time period - float val 
    theta - Heading - float val

    Returns :
    F -> Motion Model Jacobian w.r.t posterior 
    L -> Motion Model Jacobian w.r.t Noise

    """

    #  Motion Model Jacobian w.r.t Posterior

    F = np.array([[1, 0, -np.sin(theta)*delta_t*v_k_1], \
                 [0, 1, np.cos(theta)*delta_t*v_k_1], \
                 [0, 0, 1]], dtype='float') 

    # Motion Model Jacobian w.r.t Posterior
    
    L = np.array([[np.cos(theta)*delta_t, 0], \
                [np.sin(theta)*delta_t, 0], \
                [0,1]], dtype='float')  
    
    return F, L 


def normalize_angle(x):

    """
    Normalizes the angle between -pi and pi 

    Parameters :
    x -> Angle - float val 

    Returns : 
    x-> Angle - float val 

    """

    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x


def plotting_func(x_est,t):

    """
    Plotting estimated values. 
    Parameters : 
    x_est -> Estimated States - Array - (501,3)
    t -> Time array - Array - (501,)

    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_est[:, 0], x_est[:, 1],c='k')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Estimated trajectory')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t[:], x_est[:, 2],c='r')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('theta [rad]')
    ax.set_title('Estimated Theta')
    plt.show()


if __name__ == '__main__':

    """
    Main Function that processes the EKF. 
    """

    ekf = EKF_Estimation.EKF() # EKF Object 

    controls,lidar_data,variance,initial_values,t = read_data_from_pickle() #Input Data 
    Q,R,x_est,P_est = parameters_intialization() # Initializing Parameters 

    # Initializing States and Co-Variance 

    x_k_1 = x_est[0,:].reshape(3,1)
    P_k_1 = P_est[0]
    x_k_1[2,:] = normalize_angle(x_k_1[2,:])

    # Main Time loop for Prediction and Measurement 
    
    for k in range(1,len(t)):

        delta_t = t[k] - t[k-1] # Time Period 

        # Prediction Step 
        x_k_1,P_k_1 = ekf.prediction_step(delta_t,controls[0][k-1],controls[1][k-1],x_k_1,P_k_1,Q,jacobian_motion_model,motion_model)
        
        for i in range(len(lidar_data[2][k])):
            """
            Measurement Update Loop. 
            """
            x_k_1,P_k_1 = ekf.measurement_update(lidar_data[1][i],lidar_data[2][k][i],lidar_data[0][k][i],P_k_1,x_k_1,Q,R,lidar_data[3],jacobian_measurement_model)

        # Estimated States

        x_est[k,0] = x_k_1[0]
        x_est[k,1] = x_k_1[1]
        x_est[k,2] = x_k_1[2]
        P_est[k,:,:] = P_k_1

    # Plotting the estimated Values.
    plotting_func(x_est,t)






