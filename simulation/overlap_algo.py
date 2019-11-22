"""
Wireframe 3D cube simulation.
""" 

class objects:
    # gets the orthographic/axonometric projection of a single vertice
    # inputs x, y, z coordinates of midpoint; yaw, pitch, roll angles; length, width, height of the cuboid
    # outputs the projected coordiates of the vertice on the xy-plane

    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = float(x), float(y), float(z)
 
    def object2plane(self, object_name):
        object_handle = simGetObjectHandle(str(object_name))

        # get object coordinates in world reference frame
        object_centre = simGetObjectPosition(object_handle, -1)
        print(type(object_centre))
        X = object_centre[0]
        Y = object_centre[1]
        Z = object_centre[2]

        # get bounding box coordinates in object reference frame
        res_minX, minX = simGetObjectFloatParameter(object_handle, 15)
        res_minY, minY = simGetObjectFloatParameter(object_handle, 16)
        res_maxX, maxX = simGetObjectFloatParameter(object_handle, 18)
        res_maxY, maxY = simGetObjectFloatParameter(object_handle, 19)
        assert (res_minX == 1 and res_minY == 1 and res_maxX == 1 and res_maxY == 1), "simGetObjectFloatParameter gives invalid results"

        # get bounding box coordinates in world reference frame
        minX = minX + X
        minY = minY + Y
        maxX = maxX + X
        maxY = maxY + Y

        return minX, minY, maxX, maxY


    def check_overlap(self, object_main, objects_list):
        overlap_list=[]

        # returns list of objects that overlap with object_main
        main_minX, main_minY, main_maxX, main_maxY = object2plane(object_main)
        
        for obj in objects_list:
            obj_minX, obj_minY, obj_maxX, obj_maxY = object2plane(obj)
            # check if any corner of obj overlaps with object_main
            if ((obj_minX > main_minX and obj_minX < main_maxX) or (obj_maxX > main_minX and obj_maxX < main_maxX)) and ((obj_minY > main_minY and obj_minY < main_maxY) or (obj_maxY > main_minY and obj_maxY < main_maxY)):
                if obj != object_main:
                    overlap_list.append(obj)
        
        return overlap_list

