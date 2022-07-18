#!/usr/bin/env python3

import numpy as np
import math
import PyKDL as kdl
from sklearn.preprocessing import Normalizer
import pickle
import rospy
from std_msgs.msg import Header, Float64
# from beginner_tutorials.msg import NN
from visualization_msgs.msg import Marker
# from geometry_msgs.msg import Point
import time

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

aaa = 1
Done = False
solved = 0
error = 0

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

def read_save_data():
    N = pickle.load(open("/home/ayong/arm_robot/arm_ws/src/beginner_tutorials/src/N.sav", 'rb'))
    regr = pickle.load(open("/home/ayong/arm_robot/arm_ws/src/beginner_tutorials/src/model.sav", 'rb'))
    X_train_norm = pickle.load(open("/home/ayong/arm_robot/arm_ws/src/beginner_tutorials/src/X_train_norm.sav", 'rb'))
    y_train = pickle.load(open("/home/ayong/arm_robot/arm_ws/src/beginner_tutorials/src/y_train.sav", 'rb'))
    # score = regr.score(X_train_norm, y_train)
    # print("Accuracy :", score)
    return N, regr, X_train_norm, y_train

def cekError(f_target, end_effector):
    f_result = end_effector
    f_diff = f_target.Inverse() * f_result
    [dx, dy, dz] = f_diff.p
    [drz, dry, drx] = f_diff.M.GetEulerZYX()
    error = np.sqrt(dx**2 + dy**2 + dz**2 + drx**2 + dry**2 + drz**2)
    error_pos = np.sqrt(dx**2 + dy**2 + dz**2)
    error_rot = np.sqrt(drx**2 + dry**2 + drz**2)
    error_list = [dx, dy, dz, drx, dry, drz]
    return error, error_list, error_pos, error_rot

def FK(thetas, links):
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
    return end_effector, joint1, joint2, joint3, joint4, joint5, joint6

def init_publisher():
    # global rate
    # rospy.init_node('joint_positions_node', anonymous=True)
    # rate = rospy.Rate(1)
    pub1 = rospy.Publisher('/arm_robot/link1_joint_position_controller/command', Float64, queue_size=100)
    pub2 = rospy.Publisher('/arm_robot/link2_joint_position_controller/command', Float64, queue_size=100)
    pub3 = rospy.Publisher('/arm_robot/link3_joint_position_controller/command', Float64, queue_size=100)
    pub4 = rospy.Publisher('/arm_robot/link4_joint_position_controller/command', Float64, queue_size=100)
    pub5 = rospy.Publisher('/arm_robot/link5_joint_position_controller/command', Float64, queue_size=100)
    pub6 = rospy.Publisher('/arm_robot/link6_joint_position_controller/command', Float64, queue_size=100)
    return pub1, pub2, pub3, pub4, pub5, pub6

def publish_joint(pub1, pub2, pub3, pub4, pub5, pub6, joints):
    # rospy.loginfo(joints[0,0])
    # rospy.loginfo(joints[0,1])
    # rospy.loginfo(joints[0,2])
    # rospy.loginfo(joints[0,3])
    # rospy.loginfo(joints[0,4])
    # rospy.loginfo(joints[0,5])
    # print("============================================")
    # print("Target frame :")
    # print(end_effector)
    # print("============================================")
    for i in range(8):
        pub1.publish(joints[0])
        pub2.publish(joints[1])
        pub3.publish(joints[2])
        pub4.publish(joints[3])
        pub5.publish(joints[4])
        pub6.publish(joints[5])
        # rate.sleep()

def publish_marker(pose,color):
    global aaa,solved, error
    marker = Marker()
    marker.header.frame_id = "base"
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.id = aaa
    marker.scale.x = 0.035
    marker.scale.y = 0.035
    marker.scale.z = 0.035
    if color == 0:
        marker.color.r = 0
        marker.color.g = 1
        marker.color.b = 0
        solved += 1
    elif color == 1:
        marker.color.r = 1
        marker.color.g = 0
        marker.color.b = 0
        error +=1

    marker.color.a = 1.0
    
    marker.pose.position.x = pose[0]
    marker.pose.position.y = pose[1]
    marker.pose.position.z = pose[2]

    marker.pose.orientation.x = pose[3]
    marker.pose.orientation.y = pose[4]
    marker.pose.orientation.z = pose[5]
    marker.pose.orientation.w = pose[6]

    pub = rospy.Publisher('/visualization_marker', Marker, queue_size=1)
    pub.publish(marker)
    aaa+=1

def main():
    rospy.init_node('marker_basic_node', anonymous=True)
    rate = rospy.Rate(0.5)
    pub1, pub2, pub3, pub4, pub5, pub6 = init_publisher()
    N, regr, _, _ = read_save_data()
    
    n_marker = 100
    iter_ke = 0
    erroravg = 0

    for i in range(n_marker):
        #input sudut tiap joint secara random
        theta1  = np.radians(np.random.randint(-180,180))
        theta2  = np.radians(np.random.randint(-90,90))
        theta3  = np.radians(np.random.randint(-90,90))
        theta4  = np.radians(np.random.randint(-180,180))
        theta5  = np.radians(np.random.randint(-90,90))
        theta6  = np.radians(np.random.randint(-180,180))
        thetas = [theta1, theta2, theta3, theta4, theta5, theta6]

        end_effector,_,_,_,_,_,_ = FK(thetas, links)
        
        f_eff = kdl.Frame(kdl.Rotation(end_effector[0,0], end_effector[0,1], end_effector[0,2],
                                       end_effector[1,0], end_effector[1,1], end_effector[1,2],
                                       end_effector[2,0], end_effector[2,1], end_effector[2,2]),
                            kdl.Vector(end_effector[0,3], end_effector[1,3], end_effector[2,3]))

        [vx, vy, vz] = f_eff.p
        [qx, qy, qz, qw] = f_eff.M.GetQuaternion()
        X = np.array([vx, vy, vz, qx, qy, qz, qw])
        input_norm = N.transform(X.reshape(1,7))
        
        start_time = time.time()
        joints = regr.predict(input_norm)
        print("--- %s seconds ---" % (time.time() - start_time))
        # print("waktu (s): ", end - start)

        joints = [joints[0,0],joints[0,1],joints[0,2],joints[0,3],joints[0,4],joints[0,5]]
        predict_end_effector,_,_,_,_,_,_ = FK(joints, links)
        predict_f_eff = kdl.Frame(kdl.Rotation(predict_end_effector[0,0], predict_end_effector[0,1], predict_end_effector[0,2],
                                            predict_end_effector[1,0], predict_end_effector[1,1], predict_end_effector[1,2],
                                            predict_end_effector[2,0], predict_end_effector[2,1], predict_end_effector[2,2]),
                                    kdl.Vector(predict_end_effector[0,3], predict_end_effector[1,3], predict_end_effector[2,3]))

        [pvx, pvy, pvz] = predict_f_eff.p
        [pqx, pqy, pqz, pqw] = predict_f_eff.M.GetQuaternion()
        pose = np.array([pvx, pvy, pvz, pqx, pqy, pqz, pqw])
        # print("Random Position :", X)
        # print("Predict Position :", pose)
        # print("Joints in deg:", joints)

        _,_,c,d = cekError(f_eff,predict_f_eff)
        # print("error avg :", a)
        # print("error list :", b)
        # print("error pos m :", c)
        
        c = c*100
        iter_ke += 1

        publish_joint(pub1, pub2, pub3, pub4, pub5, pub6, joints)
        if c <= 1 or d <= 0.01:
            publish_marker(pose,0)
        else:
            publish_marker(pose,1)
        solvability_rate = solved/n_marker
        solvability_rate = solvability_rate*100
        print("Total marker : ", n_marker)
        print("iterasi ke : ", iter_ke)
        print("Berhasil", solved)
        print("Gagal", error)
        print("error pos :", c, "cm")
        print("error rot :", d, "rad")
        print("solvability_rate : %.0f"% solvability_rate)
        print("====================================")
        rate.sleep()
    

if __name__ == "__main__":
    main()