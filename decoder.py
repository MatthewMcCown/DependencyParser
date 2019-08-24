from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)

        while state.buffer: 
            self.input_representation = self.extractor.get_input_representation(words,pos,state)
            self.input_representation = np.reshape(self.input_representation,(-1,6))
            self.prediction = self.model.predict(self.input_representation)
            self.prediction_list = self.prediction[0].tolist()
            self.output_proabilities = []
            for i in range(0,len(self.prediction_list)):
                self.output_proabilities.append((self.prediction_list[i],i))
            self.output_proabilities.sort(reverse = True, key = lambda x: x[0])
            # Check the rules for actions

            for item in self.output_proabilities:
                self.predicted_action = self.output_labels.get(item[1])[0]
                # arc-left or arc-right are not permitted if the stack is empty.
                # Finally, the root node must never be the target of a left-arc. 
                if self.predicted_action == 'left_arc':
                    if len(state.stack) == 0 or state.stack[-1] == 0:
                        continue
                    else:
                        state.left_arc(self.output_labels.get(item[1])[1])
                        break
                elif self.predicted_action == 'right_arc':
                    if len(state.stack) == 0:
                        continue
                    else:
                        state.right_arc(self.output_labels.get(item[1])[1])
                        break
                # Shifting the only word out of the buffer is also illegal, unless the stack is empty.
                elif self.predicted_action == 'shift':
                    if len(state.buffer) == 1:
                        if len(state.stack) == 0:
                            state.shift()
                            break
                        else:
                            continue
                    else:
                        state.shift()
                        break
        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        