from conll_reader import DependencyStructure, conll_reader
from collections import defaultdict
import copy
import sys
import keras
import numpy as np

class State(object):
    def __init__(self, sentence = []):
        self.stack = []
        self.buffer = []
        if sentence: 
            self.buffer = list(reversed(sentence))
        self.deps = set() 
    
    def shift(self):
        self.stack.append(self.buffer.pop())

    def left_arc(self, label):
        self.deps.add( (self.buffer[-1], self.stack.pop(),label) )

    def right_arc(self, label):
        parent = self.stack.pop()
        self.deps.add( (parent, self.buffer.pop(), label) )
        self.buffer.append(parent)

    def __repr__(self):
        return "{},{},{}".format(self.stack, self.buffer, self.deps)

def apply_sequence(seq, sentence):
    state = State(sentence)
    for rel, label in seq:
        if rel == "shift":
            state.shift()
        elif rel == "left_arc":
            state.left_arc(label) 
        elif rel == "right_arc":
            state.right_arc(label) 
         
    return state.deps
   
class RootDummy(object):
    def __init__(self):
        self.head = None
        self.id = 0
        self.deprel = None    
    def __repr__(self):
        return "<ROOT>"

     
def get_training_instances(dep_structure):

    deprels = dep_structure.deprels
    
    sorted_nodes = [k for k,v in sorted(deprels.items())]
    state = State(sorted_nodes)
    state.stack.append(0)

    childcount = defaultdict(int)
    for ident,node in deprels.items():
        childcount[node.head] += 1
 
    seq = []
    while state.buffer: 
        if not state.stack:
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
            continue
        if state.stack[-1] == 0:
            stackword = RootDummy() 
        else:
            stackword = deprels[state.stack[-1]]
        bufferword = deprels[state.buffer[-1]]
        if stackword.head == bufferword.id:
            childcount[bufferword.id]-=1
            seq.append((copy.deepcopy(state),("left_arc",stackword.deprel)))
            state.left_arc(stackword.deprel)
        elif bufferword.head == stackword.id and childcount[bufferword.id] == 0:
            childcount[stackword.id]-=1
            seq.append((copy.deepcopy(state),("right_arc",bufferword.deprel)))
            state.right_arc(bufferword.deprel)
        else: 
            seq.append((copy.deepcopy(state),("shift",None)))
            state.shift()
    return seq   


dep_relations = ['tmod', 'vmod', 'csubjpass', 'rcmod', 'ccomp', 'poss', 'parataxis', 'appos', 'dep', 'iobj', 'pobj', 'mwe', 'quantmod', 'acomp', 'number', 'csubj', 'root', 'auxpass', 'prep', 'mark', 'expl', 'cc', 'npadvmod', 'prt', 'nsubj', 'advmod', 'conj', 'advcl', 'punct', 'aux', 'pcomp', 'discourse', 'nsubjpass', 'predet', 'cop', 'possessive', 'nn', 'xcomp', 'preconj', 'num', 'amod', 'dobj', 'neg','dt','det']


class FeatureExtractor(object):
       
    def __init__(self, word_vocab_file, pos_vocab_file):
        self.word_vocab = self.read_vocab(word_vocab_file)        
        self.pos_vocab = self.read_vocab(pos_vocab_file)        
        self.output_labels = self.make_output_labels()

    def make_output_labels(self):
        labels = []
        labels.append(('shift',None))
    
        for rel in dep_relations:
            labels.append(("left_arc",rel))
            labels.append(("right_arc",rel))
        return dict((label, index) for (index,label) in enumerate(labels))

    def read_vocab(self,vocab_file):
        vocab = {}
        for line in vocab_file: 
            word, index_s = line.strip().split()
            index = int(index_s)
            vocab[word] = index
        return vocab     

    def get_input_representation(self, words, pos, state):
        self.input_stack = state.stack
        self.input_buffer = state.buffer
        self.input_stack_length = len(state.stack)
        self.input_buffer_length = len(state.buffer)
        self.temp_stack_array = []
        self.temp_buffer_array = []
        self.np_stack_array = []
        self.null = self.word_vocab['<NULL>']

        # Test the length of the input stack and build the first part of the return array
        if  self.input_stack_length == 0:
            self.temp_stack_array.extend([self.null for i in range(3)])
        elif self.input_stack_length == 1:
            self.stack_word_1 = words[state.stack[-1]]
            self.stack_word_1_local_index = words.index(self.stack_word_1)
            self.stack_word_1_pos = pos[self.stack_word_1_local_index]
            self.stack_word_1_global_index = self.get_index(self.stack_word_1,self.stack_word_1_pos)
            self.temp_stack_array.append(self.stack_word_1_global_index)
            self.temp_stack_array.extend(self.null for i in range(2))

        elif self.input_stack_length == 2:
            self.stack_word_1 = words[state.stack[-1]]
            self.stack_word_1_local_index = words.index(self.stack_word_1)
            self.stack_word_1_pos = pos[self.stack_word_1_local_index]
            self.stack_word_1_global_index = self.get_index(self.stack_word_1,self.stack_word_1_pos)
            
            self.stack_word_2 = words[state.stack[-2]]
            self.stack_word_2_local_index = words.index(self.stack_word_2)
            self.stack_word_2_pos = pos[self.stack_word_2_local_index]
            self.stack_word_2_global_index = self.get_index(self.stack_word_2,self.stack_word_2_pos)

            self.temp_stack_array.extend([self.stack_word_1_global_index, self.stack_word_2_global_index,self.null])

        elif self.input_stack_length >= 3:
            # Word 1 on stack
            self.stack_word_1 = words[state.stack[-1]]
            self.stack_word_1_local_index = words.index(self.stack_word_1)
            self.stack_word_1_pos = pos[self.stack_word_1_local_index]
            self.stack_word_1_global_index = self.get_index(self.stack_word_1,self.stack_word_1_pos)
            # Word 2 on stack
            self.stack_word_2 = words[state.stack[-2]]
            self.stack_word_2_local_index = words.index(self.stack_word_2)
            self.stack_word_2_pos = pos[self.stack_word_2_local_index]
            self.stack_word_2_global_index = self.get_index(self.stack_word_2,self.stack_word_2_pos)
            # Word 3 on stack
            self.stack_word_3 = words[state.stack[-3]]
            self.stack_word_3_local_index = words.index(self.stack_word_3)
            self.stack_word_3_pos = pos[self.stack_word_3_local_index]
            self.stack_word_3_global_index = self.get_index(self.stack_word_3,self.stack_word_3_pos)
            self.temp_stack_array.extend([self.stack_word_1_global_index, self.stack_word_2_global_index, self.stack_word_3_global_index])

        # Test the length of the input buffer and build the first part of the return array
        if  self.input_buffer_length == 0:
            self.temp_buffer_array.extend([self.null for i in range(3)])
        elif self.input_buffer_length == 1:
            self.buffer_word_1 = words[state.buffer[-1]]
            self.buffer_word_1_local_index = words.index(self.buffer_word_1)
            self.buffer_word_1_pos = pos[self.buffer_word_1_local_index]
            self.buffer_word_1_global_index = self.get_index(self.buffer_word_1,self.buffer_word_1_pos)
            self.temp_buffer_array.append(self.buffer_word_1_global_index)
            self.temp_buffer_array.extend(self.null for i in range(2))

        elif self.input_buffer_length == 2:
            self.buffer_word_1 = words[state.buffer[-1]]
            self.buffer_word_1_local_index = words.index(self.buffer_word_1)
            self.buffer_word_1_pos = pos[self.buffer_word_1_local_index]
            self.buffer_word_1_global_index = self.get_index(self.buffer_word_1,self.buffer_word_1_pos)
            
            self.buffer_word_2 = words[state.buffer[-2]]
            self.buffer_word_2_local_index = words.index(self.buffer_word_2)
            self.buffer_word_2_pos = pos[self.buffer_word_2_local_index]
            self.buffer_word_2_global_index = self.get_index(self.buffer_word_2,self.buffer_word_2_pos)

            self.temp_buffer_array.extend([self.buffer_word_1_global_index, self.buffer_word_2_global_index,self.null])

        elif self.input_buffer_length >= 3:
            # Word 1 on buffer
            self.buffer_word_1 = words[state.buffer[-1]]
            self.buffer_word_1_local_index = words.index(self.buffer_word_1)
            self.buffer_word_1_pos = pos[self.buffer_word_1_local_index]
            self.buffer_word_1_global_index = self.get_index(self.buffer_word_1,self.buffer_word_1_pos)
            # Word 2 on buffer
            self.buffer_word_2 = words[state.buffer[-2]]
            self.buffer_word_2_local_index = words.index(self.buffer_word_2)
            self.buffer_word_2_pos = pos[self.buffer_word_2_local_index]
            self.buffer_word_2_global_index = self.get_index(self.buffer_word_2,self.buffer_word_2_pos)
            # Word 3 on buffer
            self.buffer_word_3 = words[state.buffer[-3]]
            self.buffer_word_3_local_index = words.index(self.buffer_word_3)
            self.buffer_word_3_pos = pos[self.buffer_word_3_local_index]
            self.buffer_word_3_global_index = self.get_index(self.buffer_word_3,self.buffer_word_3_pos)
            self.temp_buffer_array.extend([self.buffer_word_1_global_index, self.buffer_word_2_global_index, self.buffer_word_3_global_index])

        self.np_stack_array.extend(item for item in self.temp_stack_array)
        self.np_stack_array.extend(self.temp_buffer_array)
        self.np_stack_array = np.array(self.np_stack_array)
        return self.np_stack_array

    def get_index(self,word,word_pos):
        if word_pos is None:
            return self.word_vocab['<ROOT>']
        if word_pos == 'CD':
            return self.word_vocab['<CD>']
        if word_pos == 'NNP':
            return self.word_vocab['<NNP>']
        if word_pos == 'UNK':
            return self.word_vocab['<UNK>']
        if word_pos == 'NULL':
            return self.word_vocab['<NULL>']
        if word.lower() in self.word_vocab:
            return self.word_vocab[word.lower()]
        elif word.lower() not in self.word_vocab:
            return self.word_vocab['<UNK>']

    def get_output_representation(self, output_pair):  
        # # Construct a dict of the 91 possible output_pair combinations. 
        # # The values are integer indices incremented by 1.
        self.output_pair_int = self.output_labels.get(output_pair)
        self.return_array = keras.utils.to_categorical(self.output_pair_int, num_classes = 91)
        self.return_array = np.array(self.return_array)
        return self.return_array
    
def get_training_matrices(extractor, in_file):
    inputs = []
    outputs = []
    count = 0 
    for dtree in conll_reader(in_file): 
        words = dtree.words()
        pos = dtree.pos()
        for state, output_pair in get_training_instances(dtree):
            inputs.append(extractor.get_input_representation(words, pos, state))
            outputs.append(extractor.get_output_representation(output_pair))
        if count%100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
        count += 1
    sys.stdout.write("\n")
    return np.vstack(inputs),np.vstack(outputs)

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 


    with open(sys.argv[1],'r') as in_file:   

        extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
        print("Starting feature extraction... (each . represents 100 sentences)")
        inputs, outputs = get_training_matrices(extractor,in_file)
        print("Writing output...")
        np.save(sys.argv[2], inputs)
        np.save(sys.argv[3], outputs)
