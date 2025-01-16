import os 
import json
import music21 as m21
import numpy as np
import warnings

warnings.filterwarnings('ignore')

file_name = 'output_file.mid'
ACCEPTABLE_DURATIONS  = [
    0.25, 0.5 ,0.75, 1.0, 1.5, 2, 3, 4
]
song= m21.converter.parse(file_name)


def encode_song(song,time_step = 0.25):
    encoded_song = []
    # p = 60, d = 1.0 -> [60, "_","_","_"]
    for event in song.flat.notesAndRests:

        # handle notes
        if isinstance(event,m21.note.Note):
            symbol = event.pitch.midi # 60
            
        elif isinstance(event,m21.note.Rest):
            symbol = 'r'
            
        # convert the note/rest into time series notation
        steps = int(event.duration.quarterLength/time_step)
        for step in range(steps):
            if step == 0 :
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
                

    # cast encoded song to a list
    encoded_song = " ".join(map(str, encoded_song))
    
    return encoded_song

encoded_song = encode_song(song)

print(encoded_song)
