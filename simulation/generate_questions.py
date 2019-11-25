import numpy as np
import os, sys, json
from question_string_builder import QuestionStringBuilder

class Qusetion():
    def __init__(self, obj_exist_list):
        self.obj_exist_list = obj_exist_list    #the object exist in the scenes

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
            'book', 'bottle', 'cup', 'calculator','key','pen',
            'cube', 'keyboard', 'mouse', 'scissors', 'stapler','pc'
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
        '<AUX> there <ARTICLE> <OBJ> in the <TABLE>?',
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

   
        self.ent_queue = None
        self.q_str_builder = QuestionStringBuilder()
        self.question_outputJson = os.path.abspath('../questions/question.json')
        self.vocab_outputJson = os.path.abspath("../questions/vocab.json")



    def clearQueue(self):
        self.ent_queue = None

    def createQueue(self):
        all_qns = self.queryExist()
        print(all_qns)
        json.dump(all_qns, open(self.question_outputJson, 'w'))



    def queryExist(self):
        qns = []
        for obj in self.blacklist_objects['exist']:
            if obj not in self.obj_exist_list:
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
            q_str = self.q_str_builder.prepareString(q_str, object_name, 'desk')
            return {
                'obj':
                object_name,
                'question':
                q_str,
                'answer':
                a_str,
                'type':
                q_type,
            }


    def tokenize(self,seq,delim=' ',punctToRemove=None,addStartToken=True,addEndToken=True):

        if punctToRemove is not None:
            for p in punctToRemove:
                seq = str(seq).replace(p, '')

        tokens = str(seq).split(delim)
        if addStartToken:
            tokens.insert(0, '<START>')

        if addEndToken:
            tokens.append('<END>')

        return tokens


    def buildVocab(self,sequences,
               minTokenCount=1,
               delim=' ',
               punctToRemove=None,
               addSpecialTok=False):
        SPECIAL_TOKENS = {
            '<NULL>': 0,
            '<START>': 1,
            '<END>': 2,
            '<UNK>': 3,
        }

        tokenToCount = {}
        for seq in sequences:
            seqTokens = self.tokenize(seq,delim=delim,punctToRemove=punctToRemove,addStartToken=False,addEndToken=False)
            for token in seqTokens:
                if token not in tokenToCount:
                    tokenToCount[token] = 0
                tokenToCount[token] += 1

        tokenToIdx = {}
        if addSpecialTok == True:
            for token, idx in SPECIAL_TOKENS.items():
                tokenToIdx[token] = idx
        for token, count in sorted(tokenToCount.items()):
            if count >= minTokenCount:
                tokenToIdx[token] = len(tokenToIdx)

        return tokenToIdx



    def encode(self,seqTokens, tokenToIdx, allowUnk=False):
        seqIdx = []
        for token in seqTokens:
            if token not in tokenToIdx:
                if allowUnk:
                    token = '<UNK>'
                else:
                    raise KeyError('Token "%s" not in vocab' % token)
            seqIdx.append(tokenToIdx[token])
        return seqIdx


    def decode(self,seqIdx, idxToToken, delim=None, stopAtEnd=True):
        tokens = []
        for idx in seqIdx:
            tokens.append(idxToToken[idx])
            if stopAtEnd and tokens[-1] == '<END>':
                break
        if delim is None:
            return tokens
        else:
            return delim.join(tokens)


    def create_vocab(self):
        question_file = open(self.question_outputJson,'r',encoding='utf-8')
        questions = json.load(question_file)
        answerTokenToIdx = self.buildVocab((str(q['answer']) for q in questions
                                       if q['answer'] != 'NIL'))
        questionTokenToIdx = self.buildVocab(
            (q['question'] for q in questions if q['answer'] != 'NIL'),
            punctToRemove=['?'],
            addSpecialTok=True)

        vocab = {
            'questionTokenToIdx': questionTokenToIdx,
            'answerTokenToIdx': answerTokenToIdx,
        }
        json.dump(vocab, open(self.vocab_outputJson, 'w'))
        return vocab


    