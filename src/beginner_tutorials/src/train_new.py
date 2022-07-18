import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
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

basetoj1    = 0.1
j1toj2      = 0.1
j2toj3      = 0.16
j3toj4      = 0.089
j4toj5      = 0.105
j5toj6      = 0.021000
j6toeff     = 0.1736746
links = [basetoj1,j1toj2,j2toj3,j3toj4,j4toj5,j5toj6,j6toeff]

#m
# a1      = 0.1
# a2      = 0.1
# a3      = 0.16
# a4      = 0.089
# a5      = 0.105
# a6      = 0.021000
# eff     = 0.1736746
# links = [a1,a2,a3,a4,a5,a6]

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

def forward_kinematics(thetas):
    base = TF()
    joint1_from_base = TF(rot_axis='z', q=(thetas[0]), dz=(links[0]))
    joint1 = base.dot(joint1_from_base)
    joint2_from_joint1 = TF(rot_axis='y', q=(thetas[1]), dz=(links[1]))
    joint2 = joint1.dot(joint2_from_joint1)
    joint3_from_joint2 = TF(rot_axis='y', q=(thetas[2]), dz=(links[2]))
    joint3 = joint2.dot(joint3_from_joint2)
    joint4_from_joint3 = TF(rot_axis='z', q=(thetas[3]), dz=(links[3]))
    joint4 = joint3.dot(joint4_from_joint3)
    joint5_from_joint4 = TF(rot_axis='y', q=(thetas[4]), dz=(links[4]))
    joint5 = joint4.dot(joint5_from_joint4)
    joint6_from_joint5 = TF(rot_axis='z', q=(thetas[5]), dz=(links[5]))
    joint6 = joint5.dot(joint6_from_joint5)
    end_effector_from_joint6 = TF(dz=links[6])
    end_effector = joint6.dot(end_effector_from_joint6)

    f_eff = kdl.Frame(kdl.Rotation(end_effector[0,0], end_effector[0,1], end_effector[0,2],
                                   end_effector[1,0], end_effector[1,1], end_effector[1,2],
                                   end_effector[2,0], end_effector[2,1], end_effector[2,2]),
                        kdl.Vector(end_effector[0,3], end_effector[1,3], end_effector[2,3]))

    x, y, z = f_eff.p
    qx, qy, qz, qw = f_eff.M.GetQuaternion()
    return x, y, z, qx, qy, qz, qw
    

def main():
    start_time = time.time()
    X, y = make_regression(n_samples=50000, n_features=13, n_targets=6)

    for i in range(X.shape[0]):
        #input sudut tiap joint secara random
        thetas = [np.random.uniform(-np.pi,np.pi),
                  np.random.uniform(-np.pi,np.pi),
                  np.random.uniform(-np.pi,np.pi),
                  np.random.uniform(-np.pi,np.pi),
                  np.random.uniform(-np.pi,np.pi),
                  np.random.uniform(-np.pi,np.pi)]

        delta_q = np.random.normal(loc=[0, 0, 0, 0, 0, 0], scale=[0.33, 0.33, 0.33, 0.33, 0.33, 0.33], size=None)
        q_current = np.clip((thetas + delta_q), a_min=[-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi], a_max=[np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])

        xi, yi, zi, qxi, qyi, qzi, qwi = forward_kinematics(thetas)
        xc, yc, zc, qxc, qyc, qzc, qwc = forward_kinematics(q_current)

        pose_init = np.array([xi, yi, zi, qxi, qyi, qzi, qwi])
        pose_current = np.array([xc, yc, zc, qxc, qyc, qzc, qwc])
        delta_pose = pose_current - pose_init

        X[i,:] = np.hstack((thetas, delta_pose))
        y[i,:] = delta_q

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    N = Normalizer().fit(X_train)
    # X_train_norm = N.transform(X_train)
    print("SEDANG TRAIN DATA...")
    print("JANGAN DI CLOSE!")
    regr = MLPRegressor(hidden_layer_sizes=(4096,4096,4096,4096,4096), max_iter=1000,random_state=1).fit(X_train, y_train)
    print("TRAIN DATA SELESAI")
    score = regr.score(X_train, y_train)
    print("Score : ", score)
    print("===================================")
    print("TESTING PREDICT")
    print("y_actual ", y_test[5])
    y_pred = regr.predict(X_test[5].reshape(1,13))
    print("y_pred : ", y_pred)
    print("===================================")
    y_test_pred = regr.predict(X_test)
    print("validation data : ", y_test_pred.shape)
    mse = mean_squared_error(y_test, y_test_pred)
    print("mse  : ", mse)
    rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    print("rmse : ", rmse)
    mae = mean_absolute_error(y_test, y_test_pred)
    print("mae  : ", mae)

    pickle.dump(N, open("N.sav", 'wb'))
    pickle.dump(regr, open("model.sav", 'wb'))
    pickle.dump(X_train, open("X_train_norm.sav", 'wb'))
    pickle.dump(y_train, open("y_train.sav", 'wb'))
    print("===================================")
    print("Lama train :")
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
