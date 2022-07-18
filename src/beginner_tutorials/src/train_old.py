import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import math
import PyKDL as kdl
from sklearn.datasets import make_regression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

np.random.seed(1)

#m
# a1      = 0.098
# a2      = 0.16
# a3      = 0.0905
# a4      = 0.1045
# a5      = 0
# a6      = 0.07765

#m
a1      = 0.1
a2      = 0.1
a3      = 0.16
a4      = 0.089
a5      = 0.105
a6      = 0.021000
links = [a1,a2,a3,a4,a5,a6]

def RX(yaw):
    return np.array([[1, 0, 0], 
                     [0, math.cos(yaw), -math.sin(yaw)], 
                     [0, math.sin(yaw), math.cos(yaw)]])   

def RY(delta):
    return np.array([[math.cos(delta), 0, math.sin(delta)], 
                     [0, 1, 0], 
                     [-math.sin(delta), 0, math.cos(delta)]])

def RZ(theta):
    return np.array([[math.cos(theta), -math.sin(theta), 0], 
                     [math.sin(theta), math.cos(theta), 0], 
                     [0, 0, 1]])

def TF(rot_axis=None, q=0, dx=0, dy=0, dz=0):
    if rot_axis == 'x':
        R = RX(q)
    elif rot_axis == 'y':
        R = RY(q)
    elif rot_axis == 'z':
        R = RZ(q)
    elif rot_axis == None:
        R = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
    
    T = np.array([[R[0,0], R[0,1], R[0,2], dx],
                  [R[1,0], R[1,1], R[1,2], dy],
                  [R[2,0], R[2,1], R[2,2], dz],
                  [0, 0, 0, 1]])
    return T

def main():
    start_time = time.time()
    X, y = make_regression(n_samples=10000, n_features=7, n_targets=6)

    for i in range(10000):
        #input sudut tiap joint secara random
        theta1  = np.radians(np.random.randint(-180,180))
        theta2  = np.radians(np.random.randint(-90,90))
        theta3  = np.radians(np.random.randint(-90,90))
        theta4  = np.radians(np.random.randint(-180,180))
        theta5  = np.radians(np.random.randint(-90,90))
        theta6  = np.radians(np.random.randint(-180,180))
        thetas = [theta1, theta2, theta3, theta4, theta5, theta6]
        
        joint1 = TF(rot_axis='z', q=(thetas[0]))
        joint2_from_joint1 = TF(rot_axis='y', q=(thetas[1]), dz=(links[0]))
        joint2 = joint1.dot(joint2_from_joint1)
        joint3_from_joint2 = TF(rot_axis='y', q=(thetas[2]),dz=(links[1]))
        joint3 = joint2.dot(joint3_from_joint2)
        joint4_from_joint3 = TF(rot_axis='z', q=(thetas[3]), dy=-0.053, dz=(links[2]))
        joint4 = joint3.dot(joint4_from_joint3)
        joint5_from_joint4 = TF(rot_axis='y', q=(thetas[4]), dy=0.053, dz=(links[3]))
        joint5 = joint4.dot(joint5_from_joint4)
        joint6_from_joint5 = TF(rot_axis='z', q=(thetas[5]), dy=-0.015, dz=(links[4]))
        joint6 = joint5.dot(joint6_from_joint5)
        end_effector_from_joint6 = TF(dy=0.015000, dz=links[5])
        end_effector = joint6.dot(end_effector_from_joint6)

        f_eff = kdl.Frame(kdl.Rotation(end_effector[0,0], end_effector[0,1], end_effector[0,2],
                                       end_effector[1,0], end_effector[1,1], end_effector[1,2],
                                       end_effector[2,0], end_effector[2,1], end_effector[2,2]),
                            kdl.Vector(end_effector[0,3], end_effector[1,3], end_effector[2,3]))

        [vx, vy, vz] = f_eff.p
        [qx, qy, qz, qw] = f_eff.M.GetQuaternion()

        X[i] = np.array([vx, vy, vz, qx, qy, qz, qw])
        y[i] = np.array([theta1, theta2, theta3, theta4, theta5, theta6])

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    N = Normalizer().fit(X_train)
    X_train_norm = N.transform(X_train)
    print("TRAIN DATA IS RUNNING...")
    regr = MLPRegressor(hidden_layer_sizes=(3000,3000,3000,3000,3000), max_iter=3000, random_state=1).fit(X_train_norm, y_train)
    print("TRAIN DATA IS DONE!")
    print("===================================")
    print("CHECKING SCORE OF THE RESULT...")
    score = regr.score(X_train_norm, y_train)
    print("Score : ", score)
    print("===================================")

    X_test_norm = N.transform(X_test)
    y_test_pred = regr.predict(X_test_norm)
    print("validation data : ", y_test_pred.shape)

    mse = mean_squared_error(y_test, y_test_pred)
    print("mse : ", mse)
    rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    print("rmse : ", rmse)
    mae = mean_absolute_error(y_test, y_test_pred)
    print("mae : ", mae)

    pickle.dump(N, open("N.sav", 'wb'))
    pickle.dump(regr, open("model.sav", 'wb'))
    pickle.dump(X_train_norm, open("X_train_norm.sav", 'wb'))
    pickle.dump(y_train, open("y_train.sav", 'wb'))
    print("===================================")
    print("DONE!")
    print("Lama train :")
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()