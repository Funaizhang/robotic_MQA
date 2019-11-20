
try:
    from vrep import*
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

import time
import os
import sys
import numpy.random as random
import numpy as np
import math
from collections import defaultdict
#import PIL.Image as Image
import array
#import cv2 as cv

class Camera(object):
    """
        # kinect camera in simulation
    """
    def __init__(self, clientID):
        """
            Initialize the Camera in simulation
        """
        self.RAD2EDG = 180 / math.pi
        self.EDG2RAD = math.pi / 180
        self.Save_IMG = True
        self.Save_PATH_COLOR = r'./color'
        self.Save_PATH_DEPTH = r'./depth'
        self.Dis_FAR = 10
        self.depth_scale = 1000
        self.Img_WIDTH = 512
        self.Img_HEIGHT = 424
        self.border_pos = [120,375,100,430]# [68,324,112,388] #up down left right of the box
        self.theta = 70
        self.Camera_NAME = r'kinect'
        self.Camera_RGB_NAME = r'kinect_rgb'
        self.Camera_DEPTH_NAME = r'kinect_depth'
        self.clientID = clientID
        self._setup_sim_camera()
        self._mkdir_save(self.Save_PATH_COLOR)
        self._mkdir_save(self.Save_PATH_DEPTH)

    def _mkdir_save(self, path_name):
        if not os.path.isdir(path_name):         
            os.mkdir(path_name)

    def _euler2rotm(self,theta):
        """
            -- Get rotation matrix from euler angles
        """
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                        [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                        ])
        R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                        [0,                     1,      0                   ],
                        [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                        ])         
        R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                        [math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])            
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R


    def _setup_sim_camera(self):
        """
            -- Get some param and handles from the simulation scene
            and set necessary parameter for camera
        """
        # Get handle to camera
        _, self.cam_handle = simxGetObjectHandle(self.clientID, self.Camera_NAME, simx_opmode_oneshot_wait)
        _, self.kinectRGB_handle = simxGetObjectHandle(self.clientID, self.Camera_RGB_NAME, simx_opmode_oneshot_wait)
        _, self.kinectDepth_handle = simxGetObjectHandle(self.clientID, self.Camera_DEPTH_NAME, simx_opmode_oneshot_wait)
        # Get camera pose and intrinsics in simulation
        _, self.cam_position = simxGetObjectPosition(self.clientID, self.cam_handle, -1, simx_opmode_oneshot_wait)
        _, cam_orientation = simxGetObjectOrientation(self.clientID, self.cam_handle, -1, simx_opmode_oneshot_wait)

        self.cam_trans = np.eye(4,4)
        self.cam_trans[0:3,3] = np.asarray(self.cam_position)
        self.cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
        self.cam_rotm = np.eye(4,4)
        self.cam_rotm[0:3,0:3] = np.linalg.inv(self._euler2rotm(cam_orientation))
        self.cam_pose = np.dot(self.cam_trans, self.cam_rotm) # Compute rigid transformation representating camera pose
        self._intri_camera()

    def _intri_camera(self):  #the paramter of camera
        """
            Calculate the intrinstic parameters of camera
        """
        # ref: https://blog.csdn.net/zyh821351004/article/details/49786575
        fx = -self.Img_WIDTH/(2.0 * math.tan(self.theta * self.EDG2RAD / 2.0))
        fy = fx
        u0 = self.Img_HEIGHT/ 2
        v0 = self.Img_WIDTH / 2
        self.intri = np.array([[fx, 0, u0],
                               [0, fy, v0],
                               [0, 0, 1]])


    def get_camera_data(self):
        """
            -- Read images data from vrep and convert into np array
        """
        # Get color image from simulation
        res, resolution, raw_image = simxGetVisionSensorImage(self.clientID, self.kinectRGB_handle, 0, simx_opmode_oneshot_wait)
        # self._error_catch(res)
        color_img = np.array(raw_image, dtype=np.uint8)
        color_img.shape = (resolution[1], resolution[0], 3)
        color_img = color_img.astype(np.float)/255
        color_img[color_img < 0] += 1
        color_img *= 255
        color_img = np.flipud(color_img)
        color_img = color_img.astype(np.uint8)

        # Get depth image from simulation
        res, resolution, depth_buffer = simxGetVisionSensorDepthBuffer(self.clientID, self.kinectDepth_handle, simx_opmode_oneshot_wait)
        # self._error_catch(res)
        depth_img = np.array(depth_buffer)
        depth_img.shape = (resolution[1], resolution[0])
        depth_img = np.flipud(depth_img)
        depth_img[depth_img < 0] = 0
        depth_img[depth_img > 1] = 0.9999
        depth_img = depth_img * self.Dis_FAR * self.depth_scale
        self.cur_depth = depth_img
        return depth_img, color_img

    def save_image(self, cur_depth, cur_color, img_idx):
        """
            -- Save Color&Depth images
        """
        img = Image.fromarray(cur_color.astype('uint8')).convert('RGB')
        img_path = os.path.join(self.Save_PATH_COLOR, str(img_idx) + '_Rgb.png')
        img.save(img_path)
        depth_img = Image.fromarray(cur_depth.astype(np.uint32),mode='I')
        depth_path = os.path.join(self.Save_PATH_DEPTH, str(img_idx) + '_Depth.png')
        depth_img.save(depth_path)

        return depth_path, img_path

    def _error_catch(self, res):
        """
            -- Deal with error unexcepted
        """
        if res == simx_return_ok:
            print ("--- Image Exist!!!")
        elif res == simx_return_novalue_flag:
            print ("--- No image yet")
        else:
            print ("--- Error Raise")


    def pixel2ur5(self, u, v, ur5_position, push_depth, depth = 0.0, is_dst = True):
        """
            from pixel u,v and correspondent depth z -> coor in ur5 coordinate (x,y,z)
        """
        if is_dst == False:
            depth = self.cur_depth[int(u)][int(v)] / self.depth_scale

        x = depth * (u - self.intri[0][2]) / self.intri[0][0]
        y = depth * (v - self.intri[1][2]) / self.intri[1][1]
        camera_coor = np.array([x, y, depth - push_depth])
        """
            from camera coor to ur5 coor
            Notice the camera faces the plain directly and we needn't convert the depth to real z
        """
        camera_coor[2] = - camera_coor[2]
        location = camera_coor + self.cam_position - np.asarray(ur5_position)
        return location, depth

class UR5(object):
    def __init__(self, is_testing = 0 ,testing_file='test-10-obj-01.txt'):
        #test
        self.is_testing = is_testing
        self.testing_file = testing_file
        self.obj_type =[]
        self.targetPosition = np.zeros(3,dtype = np.float)
        self.targetQuaternion = np.array([0.707,0,0.707,0])
        self.baseName = r'UR5'
        table_file = os.path.abspath('../mesh/tables/tables.txt')
        file = open(table_file, 'r')
        file_content = file.readlines()
        file.close()
        self.desk_para = file_content[0].split()    
        self.workspace_limits = np.asarray([[float(self.desk_para[0]), float(self.desk_para[1])], [float(self.desk_para[2]), float(self.desk_para[3])] ])
        self.drop_height = float(self.desk_para[4])+0.2
        self.color_space = np.asarray([[78.0, 121.0, 167.0],  # blue
                                       [89.0, 161.0, 79.0],  # green
                                       [156, 117, 95],  # brown
                                       [242, 142, 43],  # orange
                                       [237.0, 201.0, 72.0],  # yellow
                                       [186, 176, 172],  # gray
                                       [255.0, 87.0, 89.0],  # red
                                       [176, 122, 161],  # purple
                                       [118, 183, 178],  # cyan
                                       [255, 157, 167]]) / 255.0  # pink
        # Read files in object mesh directory
        self.test_file_dir = os.path.abspath('test-cases/')
        self.test_preset_file = os.path.join(self.test_file_dir, self.testing_file)
        self.obj_mesh_dir=os.path.abspath('../mesh/convex')
        self.num_obj = 5
        self.mesh_list = os.listdir(self.obj_mesh_dir)
        # Randomly choose objects to add to scene
        self.obj_mesh_ind = np.random.choice(a=len(self.mesh_list), size=self.num_obj, replace=False)
        self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)) % 10, :]
        print (self.obj_mesh_ind)    #random objects

        simxFinish(-1)  # just in case, close all opened connections
        self.clientID = simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP
        if self.clientID != -1:
            print ('Connected to remote API server')
            # If testing, read object meshes and poses from test case file
            if self.is_testing:
                file = open(self.test_preset_file, 'r')
                file_content = file.readlines()
                self.test_obj_mesh_files = []
                self.test_obj_type = []
                self.test_obj_mesh_colors = []
                self.test_obj_positions = []
                self.test_obj_orientations = []
                for i in range(self.num_obj):
                    file_content_curr_object = file_content[i].split()
                    self.test_obj_mesh_files.append(os.path.join(self.obj_mesh_dir, file_content_curr_object[0]))
                    self.test_obj_type.append(file_content_curr_object[0])
                    self.test_obj_mesh_colors.append(
                        [float(file_content_curr_object[1]), float(file_content_curr_object[2]),
                         float(file_content_curr_object[3])])
                    self.test_obj_positions.append(
                        [float(file_content_curr_object[4]), float(file_content_curr_object[5]),
                         float(file_content_curr_object[6])])
                    self.test_obj_orientations.append(
                        [float(file_content_curr_object[7]), float(file_content_curr_object[8]),
                         float(file_content_curr_object[9])])
                file.close()
                self.obj_mesh_color = np.asarray(self.test_obj_mesh_colors)
            simxStartSimulation(self.clientID, simx_opmode_blocking)
            #self.add_objects()
        else:
            print ('Failed connecting to remote API server')
        _, self.ur5_handle = simxGetObjectHandle(self.clientID,self.baseName,simx_opmode_oneshot_wait)
        _, self.ur5_position = simxGetObjectPosition(self.clientID,self.ur5_handle,-1,simx_opmode_oneshot_wait)
        self.add_objects()
        self.ankleinit()


    def ankleinit(self):
        """
            # initial the ankle angle for ur5
        """
        simxSynchronousTrigger(self.clientID) 
        simxPauseCommunication(self.clientID, True)
        simxSetIntegerSignal(self.clientID, 'ICECUBE_0', 11, simx_opmode_oneshot)
        simxPauseCommunication(self.clientID, False)
        simxSynchronousTrigger(self.clientID)
        simxGetPingTime(self.clientID)
        # pause for 1s
        time.sleep(1)


    def disconnect(self):
        """
            # disconnect from v-rep
            # and stop simulation
        """
        simxStopSimulation(self.clientID,simx_opmode_oneshot)
        simxFinish(self.clientID)
        print ('Simulation ended!')

    def get_clientID(self):
        return self.clientID
        

    def ur5push(self, move_begin, move_to):
        """
            The action of the ur5 in a single push action including:
            Get to push beginning
            Push to the destination
            Return to the init pose
        """
        self.break_condition(1) 

        time.sleep(1)       
        self.ur5moveto(move_begin)
        time.sleep(0.5)
        self.ur5moveto(move_to)
        time.sleep(0.5)

        # Return to the initial pose
        self.ankleinit()


    def ur5suction(self, suction_point):
        """
            The action of the ur5 in a single suction action including:
            Get to suction_point
            Suck the object
            Return to the init pose with the object sucked
        """
        self.ur5moveto(suction_point)
        time.sleep(1)
        self.break_condition(0)
        time.sleep(1)
        # suction automatically done when suction pad is close to obj
        # Return to the initial pose with the object
        self.ankleinit()


    def ur5loose(self, suction_point):
        """
            The action of the ur5 in a single release action including:
            Get to suction_point
            Release the object
            Return to the init pose
        """

        self.ur5moveto(suction_point)       
        time.sleep(1)
        #self.break_condition(1)
        # release
        self.break_condition(1)
        time.sleep(1)
        # Return to the initial pose with the object
        self.ankleinit()


    def ur5moveto(self, dst_location):
        """
            Push the ur5 hand to the location of dst_location
        """
        simxSynchronousTrigger(self.clientID)
        self.targetPosition = dst_location
        simxPauseCommunication(self.clientID, True)
        simxSetIntegerSignal(self.clientID, 'ICECUBE_0', 21, simx_opmode_oneshot)
        for i in range(3):
            simxSetFloatSignal(self.clientID, 'ICECUBE_'+str(i+1),self.targetPosition[i],simx_opmode_oneshot)
        for i in range(4):
            simxSetFloatSignal(self.clientID, 'ICECUBE_'+str(i+4),self.targetQuaternion[i], simx_opmode_oneshot)
        simxPauseCommunication(self.clientID, False)
        simxSynchronousTrigger(self.clientID)
        simxGetPingTime(self.clientID)

    def break_condition(self, state):
        """
           set break_condition
        """
        simxSynchronousTrigger(self.clientID)
        simxPauseCommunication(self.clientID, True)
        simxSetIntegerSignal(self.clientID, 'BREAK', state, simx_opmode_oneshot)
        simxPauseCommunication(self.clientID, False)
        simxSynchronousTrigger(self.clientID)
        simxGetPingTime(self.clientID)

        

    def add_objects(self):
        # Add objects to robot workspace at x,y location and orientation
        self.object_handles = []
        self.object_type = []
        file = open(self.test_preset_file, 'w')
        for i in range(self.num_obj):
            object_idx = self.obj_mesh_ind[i]
            self.object_type.append(self.mesh_list[object_idx])
            object_color = [self.obj_mesh_color[i][0], self.obj_mesh_color[i][1], self.obj_mesh_color[i][2]]
            curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[object_idx])
            drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
            drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
            object_position = [drop_x, drop_y, self.drop_height]
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]

            curr_shape_name = 'shape'+str(i)
            if self.is_testing:
                self.object_type.append(self.test_obj_type[i])
                curr_mesh_file = self.test_obj_mesh_files[i]
                object_color= [self.test_obj_mesh_colors[i][0], self.test_obj_mesh_colors[i][1], self.test_obj_mesh_colors[i][2]]
                object_position = [self.test_obj_positions[i][0], self.test_obj_positions[i][1], self.test_obj_positions[i][2]]
                object_orientation = [self.test_obj_orientations[i][0], self.test_obj_orientations[i][1], self.test_obj_orientations[i][2]]

            print (object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name])
            ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = simxCallScriptFunction(self.clientID, 'remoteApiCommandServer',sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), simx_opmode_blocking)
            time.sleep(1)
            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                print (ret_strings)
                exit()
            curr_shape_handle = ret_ints[0]
            self.object_handles.append(curr_shape_handle)
            # create new scene  
            file_write_content = object_color+ object_position+ object_orientation
            file.write(self.object_type[i]+' ')
            for data in file_write_content:
                file.write(str(data)+' ')
            file.write('\n')
        file.close()

    def get_obj_positions_and_orientations(self):
        obj_dict = defaultdict(dict)
        print(self.object_handles)
        for object_handle in self.object_handles:

            # get object centre coordinates in world reference frame
            sim_ret, object_position = simxGetObjectPosition(self.clientID, object_handle, -1, simx_opmode_blocking)
            # print('debug --- {} {}'.format(sim_ret, object_position))
            assert (sim_ret == 0), "simxGetObjectPosition gives invalid results"
            X = object_position[0]
            Y = object_position[1]
            Z = object_position[2]

            sim_ret, object_orientation = simxGetObjectOrientation(self.clientID, object_handle, -1, simx_opmode_blocking)
            assert (sim_ret == 0), "simxGetObjectOrientation gives invalid results"

            # get bounding box coordinates in object reference frame
            sim_ret_minX, minX = simxGetObjectFloatParameter(self.clientID, object_handle, 15, simx_opmode_blocking)
            sim_ret_minY, minY = simxGetObjectFloatParameter(self.clientID, object_handle, 16, simx_opmode_blocking)
            sim_ret_maxX, maxX = simxGetObjectFloatParameter(self.clientID, object_handle, 18, simx_opmode_blocking)
            sim_ret_maxY, maxY = simxGetObjectFloatParameter(self.clientID, object_handle, 19, simx_opmode_blocking)
            assert (sim_ret_minX == 0 and sim_ret_minY == 0 and sim_ret_maxX == 0 and sim_ret_maxY == 0), "simxGetObjectFloatParameter gives invalid results"
            sizes = (maxX-minX, maxY-minY)
            print('-- debug -- obj {} size {}'.format(object_handle, sizes))
            # calculate bounding box coordinates in world reference frame
            minX = minX + X
            minY = minY + Y
            maxX = maxX + X
            maxY = maxY + Y

            obj_dict[object_handle]['position'] = object_position
            obj_dict[object_handle]['orientation'] = object_orientation
            obj_dict[object_handle]['bound'] = (minX, minY, maxX, maxY)

            print('get_obj_positions_and_orientations found {} at {} bound {}'.format(object_handle, obj_dict[object_handle]['position'], obj_dict[object_handle]['bound']))

        return obj_dict

    
    def check_overlap(self, obj_target_handle, obj_dict):
        # find the bound of the obj_target
        target_minX, target_minY, target_maxX, target_maxY = obj_dict[obj_target_handle]['bound']
        overlap_list = []

        for obj in obj_dict:
            # check if the ith obj we are looking at is obj_target
            if obj == obj_target_handle:
                continue
            # a different obj
            else:
                (obj_minX, obj_minY, obj_maxX, obj_maxY) = obj_dict[obj]['bound']

                # check if any corner of obj overlaps with obj_target
                obj_in_target = ((obj_minX > target_minX and obj_minX < target_maxX) or (obj_maxX > target_minX and obj_maxX < target_maxX)) and ((obj_minY > target_minY and obj_minY < target_maxY) or (obj_maxY > target_minY and obj_maxY < target_maxY))
                target_in_obj = ((target_minX > obj_minX and target_minX < obj_maxX) or (target_maxX > obj_minX and target_maxX < obj_maxX)) and ((target_minY > obj_minY and target_minY < obj_maxY) or (target_maxY > obj_minY and target_maxY < obj_maxY))
                if obj_in_target or target_in_obj:
                    overlap_list.append(obj)
        
        # obj_dict[obj_target_handle]['overlaps'] = overlap_list
        return overlap_list

    

class Environment(object):
    """
        # simulation environment 
    """
    def __init__(self):
        # initial the ur5 arm in simulation
        self.ur5 = UR5()
        self.ur5.ankleinit()
        self.ur5_location = self.ur5.ur5_position
        # initial the camera in simulation
        self.clientID = self.ur5.get_clientID()
        self.camera = Camera(self.clientID)
        print('\n [*] Initialize the simulation environment')



    def UR5_action(self,action):
        action_np = np.array(action)
        action_degree = action_np.shape[1]
        if action_degree == 2:
            push_depth=0.01
            start_point = action[0]
            end_point = action[1]
            move_begin, src_depth = self.camera.pixel2ur5(start_point[0], start_point[1], self.ur5_location, push_depth, is_dst = False)
            move_to, _ = self.camera.pixel2ur5(end_point[0], end_point[1], self.ur5_location, push_depth, src_depth, is_dst = True)
            self.ur5.ur5push(move_begin, move_to)
            print('\n -- Push from {} to {}' .format(start_point,end_point))
            return move_begin, move_to
        elif action_degree == 1:
            move_begin = []
            suck_point = action
            move_to, src_depth = self.camera.pixel2ur5(suck_point[0], suck_point[1], self.ur5_location, 0, is_dst = False)
            self.ur5.ur5push(move_begin, move_to)




    def close(self):
        """
            End the simulation
        """
        self.ur5.disconnect()

