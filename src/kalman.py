# Note: The code in this file is based on the code provided by:
# https://automaticaddison.com/extended-kalman-filter-ekf-with-python-code-example/


import numpy as np
from src.computerVision import findThymio



IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080


def inverseSpeedConversion(vR, vL, R, L, Cr, Cl):
    v = (R/2)*(vR/Cr + vL/Cl)
    omega = (R/L)*(vR/Cr - vL/Cl)
    return v, omega

def speedConverison(v,omega,R,L,Cr,Cl):
    v_r = ((2*v + omega*L)/(2*R))*Cr
    v_l = ((2*v - omega*L)/(2*R))*Cl
    return v_r, v_l


R = 20 #mm
L = 105 #mm
Cr = 69.33821285
Cl = 70.74580573
#varv = 0.0030079585145924904
#varomega = 1.807798653077301e-05
varv = 2.44539
varomega = 0.0011476
 
# State matrix A

A_k_minus_1 = np.array([[1.0,0,0,0,0],[0,1.0,0,0,0],[0,0,1.0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
 
# Input matrix B
def getB(yaw, deltak):
    B = np.array([ [np.cos(yaw)*deltak, 0],[np.sin(yaw)*deltak, 0],[0, deltak],[1, 0],[0, 1]])
    return B

#Jacobian of A
def getJacobainA(yaw, deltak, v):
    jacobianA = np.array([
        [1.0, 0.0, -deltak * v * np.sin(yaw), deltak * np.cos(yaw), 0.0],
        [0.0, 1.0, deltak * v * np.cos(yaw), deltak * np.sin(yaw), 0.0],
        [0.0, 0.0, 1.0, 0.0, deltak],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0]
        ])

    return jacobianA


# Process noise w
process_noise_w_k_minus_1 = np.array([0,0,0,0,0])
     
# State model noise covariance matrix Q_k
qx = 10 
qy = 10 
qgamma = 0.17454 
qv = varv
qomega = varomega                
Q_k = np.array([[qx, 0, 0,0,0],
                [0, qy, 0, 0, 0],
                [0, 0, qgamma, 0, 0],
                [0, 0, 0, qv, 0],
                [0, 0, 0, 0, qomega]])                 

# Measurement matrix H_k
H_k = np.array([[1.0,0,0,0,0],[0,1.0,0,0,0],[0,0,1.0,0,0],[0,0,0,1.0,0],[0,0,0,0,1.0]])
                      
# Sensor measurement noise covariance matrix R_k
r11 = 2
r22 = 2
r33 = 0.0436 
r44 = varv
r55 = varomega
R_k = np.array([[r11,0,0,0,0],[0,r22,0,0,0],[0,0,r33,0,0],[0,0,0,r44,0],[0,0,0,0,r55]])

# Sensor measurement noise covariance matrix R_k_nc
#R_k_nc = np.array([[np.inf,np.inf,np.inf],[np.inf,np.inf,np.inf],[np.inf,np.inf,np.inf]])
R_k_nc = np.array([[np.inf,0,0,0,0],[0,np.inf,0,0,0],[0,0,np.inf,0,0],[0,0,0,r44,0],[0,0,0,0,r55]])

                 
# Sensor noise v
sensor_noise_v_k = np.array([0,0,0,0,0])

# Initial state covariance matrix
# choose how to initialize it
P_k_minus_1 = np.array([[r11,0,0,0,0],[0,r22,0,0,0],[0,0,r33,0,0],[0,0,0,r44,0],[0,0,0,0,r55]])

                                               
                                               

#camera_vision = True means that the camera is used to estimate the position
#camera_vision = False means that the camera is covered so it is not used to estimate the position
 
def ekf(z_k_observation_vector, state_estimate_k_minus_1,control_vector_k_minus_1, P_k_minus_1, dk,camera_vision):

    # Prediction step

    #Prediction of the state estimate
    state_estimate_k = A_k_minus_1 @ (state_estimate_k_minus_1) + (getB(state_estimate_k_minus_1[2],dk)) @ (control_vector_k_minus_1) + (process_noise_w_k_minus_1)
    #print(state_estimate_k)             
    # Predicton the state covariance estimate 
    yaw = state_estimate_k[2]
    v = control_vector_k_minus_1[0]
    P_k = getJacobainA(yaw, dk, v) @ P_k_minus_1 @ getJacobainA(yaw, dk, v).T + Q_k
         
    # Update step

    measurement_residual_y_k = z_k_observation_vector - ((H_k @ state_estimate_k) + (sensor_noise_v_k))
             
    # Calculate the measurement residual covariance
    if camera_vision == True:
        S_k = H_k @ P_k @ H_k.T + R_k
    else:
        S_k = H_k @ P_k @ H_k.T + R_k_nc
         
    # Calculate the near-optimal Kalman gain

    K_k = P_k @ H_k.T @ np.linalg.pinv(S_k)
         
    # Calculate an updated state estimate for time k
    state_estimate_k = state_estimate_k + (K_k @ measurement_residual_y_k)
    #print(state_estimate_k) 
    # Update the state covariance estimate for time k
    P_k = P_k - (K_k @ H_k @ P_k)
     
    # Return the updated state and covariance estimates
    return state_estimate_k, P_k




def estimatePosition(frame,previousControlVector,dt,P_k_minus_1,estimatedState_1,measLS,measRS):

    globPosition, globAngle, __ = findThymio(frame)
    v,w = inverseSpeedConversion(measRS,measLS,R,L,Cr,Cl)

    z_k_observation_vector = np.array([globPosition[0],IMAGE_HEIGHT-globPosition[1],globAngle,v,w])

    if globPosition[0] == -1:
        try:    
            estimatedState ,P_k = ekf(z_k_observation_vector,estimatedState_1,previousControlVector,P_k_minus_1,dt,False)
        except:
            estimatedState = estimatedState_1
            P_k = P_k_minus_1
            print("did not converge")
    else:
        estimatedState ,P_k = ekf(z_k_observation_vector,estimatedState_1,previousControlVector,P_k_minus_1,dt,True)

    position = np.array([estimatedState[0],estimatedState[1]])
    angle = estimatedState[2]

    return position, angle, estimatedState, P_k
