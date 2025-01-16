import tensorflow.keras

import json
import numpy as np
import music21 as m21

from preprocess import *
MODEL_PATH = 'my_model.keras'

class Song_Generator:
    def __init__(self,model_path = MODEL_PATH):
        self.model_path = model_path
        self.model = keras.models.load_model(self.model_path)


        self.mappings = json.load(open(MAPPING_PATH,'r')) 
        
        self._start_symbols = ["/"]* SEQUENCE_LENGTH
    
    def generate(self,beginning,num_steps , max_sequence_length, selection):
        # the selection value is the measure of unpredictabilty of tune
        
        # create a song with 
        beginning = beginning.split()
        song = beginning

        beginning  = self._start_symbols + beginning

        # map beginning to integers
        beginning = [self.mappings[symbol] for symbol in beginning]
        
        for _ in range(num_steps):

            # limit the beginning to max sequence length 
            beginning = beginning[-max_sequence_length:]
            
            # OHE
            onehot_beginnning = keras.utils.to_categorical(beginning,num_classes=len(self.mappings))
            
            # onehot_beginnning= np.array(onehot_beginnning)            
            # onehot_beginnning= onehot_beginnning.resize(1,onehot_beginnning.shape)
            onehot_beginnning= onehot_beginnning[np.newaxis,...]
            
            # make prediction 
            probab = self.model.predict(onehot_beginnning)[0]
            
            output_int = self._sample_with_selection(probab,selection)
            
            # update beginning
            beginning.append(output_int)
            
            # map int to our encoding
            output_symbol = [k for k,v in self.mappings.items() if v == output_int][0]
            
            # check if we our at end of song

            if output_symbol == '/':
                break
            
            # update song
            song.append(output_symbol)
            
        return song
    
    def save_song(self,song,step_duration = 0.25 ,format='midi',file_name = 'song.mid'):

        stream = m21.stream.Stream()
        
        # convert song to note/rest
        # 55 _ _ _ r _ 60 _ 
        start_symbol = None
        step_counter = 1
        
        for i,symbol in enumerate(song):
            # handle case in which we have a note/rest
            if symbol != "_" or i+1 == len(song):
                
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == 'r':
                        m21_event = m21.note.Rest(quarterLength = quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol),quarterLength = quarter_length_duration)
                        
                    stream.append(m21_event)
                    
                    #reset the step counter 
                    step_counter = 1
                start_symbol = symbol




            # handle case in which we have a _ sign
            else:
                step_counter+=1
            
        # output to a midi file
        stream.write(format,file_name)
        
 


    def _sample_with_selection(self,prob,selection):
        prediction = np.log(prob)/selection

        # softmax
        prob = np.exp(prediction) / np.sum(np.exp(prediction)) 
        
        
        choices = range(len(prob))
        index = np.random.choice(choices, p = prob)
        
        return index

mg = Song_Generator()
seed = "60 _ 65 _ 65 _ 65 _ 67"
song = mg.generate(seed,500,SEQUENCE_LENGTH,0.9)
print(song)

mg.save_song(song)


