"""
Author - Kailash Nagarajan
Date - 3-Oct-2020
email-ID - kailashnagarajan@gmail.com 
"""

import numpy as np 
from main import normalize_angle

"""
Extended Kalman Filter - Linearizing Non-Linear Motion Models using Jacobians.
1. Predict the states using the Motion Model and Jacobian of the Motion Model.
2. Correct the states using the Kalman Gain, Sensor Values and Jacobian of the Measurement Model.
"""

class EKF:

	def __init__(self):
		pass

	def measurement_update(self,l_k,r_k,b_k,P_k_1,x_k_1,Q,R,d,jacobian):
		"""
		The Measurement Update Function Updates the predicted states using the Sensor Values and Kalman Gain.
		
		Parameters : 

		l_k -> Global Position of Landmarks of lidar in both x and y directions - List - (2,). 
		r_k -> Range  Readings of the lidar - float val.
		b_k -> Bearing Readings of the lidar - float val.
		P_k_1 -> Predicted Co-variance Matrix - Array - (3,3)
		x_k_1 -> Predicted States Matrix - Array - (3,)
		Q - Process Noise Co-Variance - Array - (2,2)
		R - Measurement Noise Co-variance - Array - (2,2)
		d - distance between robot center and lidar origin - float val
		jacobian - Function that computes the measurement jacobians (Both Noise and Last State) - Function

		Returns : 
		x_k - Corrected State - Array - (3,)
		P_k - Corrected Covariance - Array - (3,)

		"""

		# Assigning predicted state values to respective states.

		x_k = x_k_1[0] # X-position
		y_k = x_k_1[1] # Y-position
		theta_k = x_k_1[2] # Heading

		# Assigining landmark positions.

		x_l = l_k[0] # X-position of Global Landmark.

		y_l = l_k[1] # Y-position of Global Landmark.


		H_k,M_k,r,phi = jacobian(x_l,x_k,y_l,y_k,d,theta_k)

		"""
		H_k - Jacobian w.r.t last state - Array - (2,3).
		M_k - Jacobian w.r.t noise - Array - (2,2).
		r - Distance (Range) - float val.
		phi - Heading - float val.
		"""

		y_out = np.vstack([r,normalize_angle(phi)]) # Output Distance (Range) and Heading. 
		y_meas = np.vstack([r_k,normalize_angle(b_k)]) # Measured Distance (Range) and Heading.

		
		# Calculation of Kalman Gain

		K_k = P_k_1.dot(H_k.T).dot(np.linalg.inv(H_k.dot(P_k_1).dot(H_k.T) + M_k.dot(R).dot(M_k.T)))


		# State and Co-Variance Correction Step 

		x_k = x_k_1 + K_k.dot(y_meas - y_out) # State Matrix
		
		P_k = (np.identity(3)-K_k.dot(H_k)).dot(P_k_1) # Co-Variance Matrix


		return x_k,P_k 
	

	def prediction_step(self,delta_t,v,om,x_k_1,P_k_1,Q,jacobian,motion_model):

		"""
		The Prediction Update Function, Predicts the next step using the previous state and control and 
		the Motion and Noise Model.
		
		Parameters : 

		delta_t -> Time Period - float val 
		v -> Previous Linear Velocity - float val 
		om -> Previous Angular Velocity - float val 
		x_k_1 -> Current State - Array - (3,)
		P_k_1 -> Current Covariance - Array - (3,3)
		jacobian -> Function that computes the process jacobians (Both Noise and Last State) - Function
		motion_model -> Function that computes the output states for a given input states - Function 

		Returns : 
		x_k_1 : Predicted States - Array - (3,)
		P_k_1 : Predicted states - Array - (3,3)

		"""

		# Motion Model Returns the states [x,y,theta]
		x_k_1[0],x_k_1[1],x_k_1[2] = motion_model(v,om,x_k_1[0],x_k_1[1],x_k_1[2],delta_t) 
        
		#Jacobian of Motion Model w.r.t last state and Noise 
		F, L = jacobian(v,x_k_1[2],delta_t)

		# Predicted Co-Variance
		P_k_1 = F.dot((P_k_1).dot(F.T)) + L.dot((Q).dot(L.T))

		return x_k_1,P_k_1 
