import numpy as np
import os, sys, json
from question_string_builder import QuestionStringBuilder

class Qusetion():
    def __init__(self, obj_list, obj_position, obj_orientation,talble_name):
        self.table_name = table_name
        self.obj_list = obj_list
        self.obj-position = obj_position
        self.obj_orientation = obj_orientation
        self.query_fns = {
            'query_room': self.queryRoom,
            'query_count': self.queryCount,
            'query_room_count': self.queryRoomCounts,
            'query_global_object_count': self.queryGlobalObjectCounts,
            'query_room_object_count': self.queryRoomObjectCounts,
            'query_exist': self.queryExist,
            'query_logical': self.queryLogical,
            'query_color': self.queryColor,
            'query_color_room': self.queryColorRoom,
            'query_object': self.queryObject,
            'query_object_room': self.queryObjectRoom,
            'query_compare': self.queryCompare
        }

        self.blacklist_table = [
        'loggia', 'storage', 'guest room', 'hallway', 'wardrobe', 'hall',
        'boiler room', 'terrace', 'room', 'entryway', 'aeration', 'lobby',
        'office', 'freight elevator', 'passenger elevator'
        ]

        self.blacklist_objects = {
        'location': [
            'column', 'door', 'kitchen_cabinet', 'kitchen_set',
            'hanging_kitchen_cabinet', 'switch', 'range_hood_with_cabinet',
            'game_table', 'headstone', 'pillow', 'range_oven_with_hood',
            'glass', 'roof', 'cart', 'window', 'headphones_on_stand', 'coffin',
            'book', 'toy', 'workplace', 'range_hood', 'trinket', 'ceiling_fan',
            'beer', 'books', 'magazines', 'shelving', 'partition',
            'containers', 'container', 'grill', 'stationary_container',
            'bottle', 'outdoor_seating', 'stand', 'place_setting', 'arch',
            'household_appliance', 'pet', 'person', 'chandelier', 'decoration'
        ],
        'count': [
            'container', 'containers', 'stationary_container', 'switch',
            'place_setting', 'workplace', 'grill', 'shelving', 'person', 'pet',
            'chandelier', 'household_appliance', 'decoration', 'trinket',
            'kitchen_set', 'headstone', 'arch', 'ceiling_fan', 'glass', 'roof',
            'outdoor_seating', 'stand', 'kitchen_cabinet', 'coffin', 'beer',
            'book', 'books'
        ],
        'exist': [
            'container', 'containers', 'stationary_container', 'decoration',
            'trinket', 'place_setting', 'workplace', 'grill', 'switch',
            'window', 'door', 'column', 'person', 'pet', 'chandelier',
            'household_appliance', 'ceiling_fan', 'arch', 'book', 'books',
            'glass', 'roof', 'shelving', 'outdoor_seating', 'stand',
            'kitchen_cabinet', 'kitchen_set', 'coffin', 'headstone', 'beer'
        ],
        'color': [
            'container', 'containers', 'stationary_container', 'candle',
            'coffee_table', 'column', 'door', 'floor_lamp', 'mirror', 'person',
            'rug', 'sofa', 'stairs', 'outdoor_seating', 'kitchen_cabinet',
            'kitchen_set', 'switch', 'storage_bench', 'table_lamp', 'vase',
            'candle', 'roof', 'stand', 'beer', 'chair', 'chandelier',
            'coffee_table', 'column', 'trinket', 'grill', 'book', 'books',
            'curtain', 'desk', 'door', 'floor_lamp', 'hanger', 'workplace',
            'glass', 'headstone', 'kitchen_set', 'mirror', 'plant', 'shelving',
            'place_setting', 'ceiling_fan', 'stairs', 'storage_bench',
            'switch', 'table_lamp', 'vase', 'decoration', 'coffin',
            'wardrobe_cabinet', 'window', 'pet', 'cup', 'arch',
            'household_appliance'
        ],
        'color_room': [
            'column', 'door', 'kitchen_cabinet', 'kitchen_set', 'mirror',
            'household_appliance', 'decoration', 'place_setting', 'book',
            'person', 'stairs', 'switch', 'pet', 'chandelier', 'container',
            'containers', 'stationary_container', 'trinket', 'coffin', 'books',
            'ceiling_fan', 'workplace', 'glass', 'grill', 'roof', 'shelving',
            'outdoor_seating', 'stand', 'headstone', 'arch', 'beer'
        ],
        'relate': [
            'office_chair', 'column', 'door', 'switch', 'partition',
            'household_appliance', 'decoration', 'place_setting', 'book',
            'person', 'pet', 'chandelier', 'container', 'containers',
            'stationary_container', 'trinket', 'stand', 'kitchen_set', 'arch',
            'books', 'ceiling_fan', 'workplace', 'glass', 'grill', 'roof',
            'shelving', 'outdoor_seating', 'kitchen_cabinet', 'coffin',
            'headstone', 'beer'
        ],
        'dist_compare': [
            'column', 'door', 'switch', 'person', 'household_appliance',
            'decoration', 'trinket', 'place_setting', 'coffin', 'book'
            'cup', 'chandelier', 'arch', 'pet', 'container', 'containers',
            'stationary_container', 'shelving', 'stand', 'kitchen_set',
            'books', 'ceiling_fan', 'workplace', 'glass', 'grill', 'roof',
            'outdoor_seating', 'kitchen_cabinet', 'headstone', 'beer'
        ]
        }

        self.templates = {
        'location':
        'what room <AUX> the <OBJ> located in?',
        'count':
        'how many <OBJ-plural> are in the <ROOM>?',
        'room_count':
        'how many <ROOM-plural> are in the house?',
        'room_object_count':
        'how many rooms in the house have <OBJ-plural> in them?',
        'global_object_count':
        'how many <OBJ-plural> are there in all <ROOM-plural> across the house?',
        'exist':
        '<AUX> there <ARTICLE> <OBJ> in the <ROOM>?',
        'exist_logic':
        '<AUX> there <ARTICLE> <OBJ1> <LOGIC> <ARTICLE> <OBJ2> in the <ROOM>?',
        'color':
        'what color <AUX> the <OBJ>?',
        'color_room':
        'what color <AUX> the <OBJ> in the <ROOM>?',
        # prepositions of place
        'above':
        'what is above the <OBJ>?',
        'on':
        'what is on the <OBJ>?',
        'below':
        'what is below the <OBJ>?',
        'under':
        'what is under the <OBJ>?',
        'next_to':
        'what is next to the <OBJ>?',
        'above_room':
        'what is above the <OBJ> in the <ROOM>?',
        'on_room':
        'what is on the <OBJ> in the <ROOM>?',
        'below_room':
        'what is below the <OBJ> in the <ROOM>?',
        'under_room':
        'what is under the <OBJ> in the <ROOM>?',
        'next_to_room':
        'what is next to the <OBJ> in the <ROOM>?',
        # object distance comparisons
        'closer_room':
        'is the <OBJ> closer to the <OBJ> than to the <OBJ> in the <ROOM>?',
        'farther_room':
        'is the <OBJ> farther from the <OBJ> than from the <OBJ> in the <ROOM>?'
        } 

        self.debug = debug
        self.ent_queue = None
        self.q_str_builder = QuestionStringBuilder()
        self.question_outputJson = os.path.abspath('question/question.json')



    def clearQueue(self):
        self.ent_queue = None

    def  createQueue(self):
        all_qns = queryExist()
        json.dump(all_qns, open(self.question_outputJson, 'w'))



    def queryExist(self):
        qns = []
        for obj in self.blacklist_objects['exist']:
            if obj not in self.obj_list:
                qns.append(self. questionObjectBuilder(
                    'exist', obj, 'no', q_type='exist_negative'))
            else: 
                qns.append(self. questionObjectBuilder(
                    'exist', obj, 'yes', q_type='exist_positive'))
        return qns


    def questionObjectBuilder(self, template, object_name, a_str, q_type=None):
        if q_type == None:
            q_type = template

        q_str = self.templates[template]   

        if template == 'exist':
            q_str = self.q_str_builder.prepareString(q_str, object_name, self.table_name)
            return {
                'question':
                q_str,
                'answer':
                a_str,
                'type':
                q_type,
            }

       


  