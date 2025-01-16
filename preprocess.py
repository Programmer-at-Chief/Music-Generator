import os 
import json
import music21 as m21
import tensorflow.keras as keras
import numpy as np

KERN_DATASET_PATH = "Scores/Deutschl/erk"
ACCEPTABLE_DURATIONS  = [
    0.25, 0.5 ,0.75, 1.0, 1.5, 2, 3, 4
]
SAVE_DIR = "Dataset"
SINGLE_FILE = "combination"
SEQUENCE_LENGTH = 64
MAPPING_PATH = 'mapping.json'

musescore_path = r'/bin/mscore'  # Adjust for your OS
m21.environment.set('musescoreDirectPNGPath' ,musescore_path)
m21.environment.set('musicxmlPath',musescore_path)

def load_songs(path):
    songs = []
    for root,dirs,files in os.walk(path):
        for file in files:
            if file[-3:] == 'krn':
                song= m21.converter.parse(os.path.join(root,file))
                songs.append(song)
                
    return songs


def has_acceptable_durations(song,acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

def preprocess(path):
    print("Loading songs..")
    
    songs = load_songs(path)
    
    print(f"Loaded {len(songs)} songs")
    
    for i,song in enumerate(songs):
        
        # filter out songs with non acceptable durations 
        if not has_acceptable_durations(song,ACCEPTABLE_DURATIONS):
            print('Not acceptable')
            
        # transpose songs to Cmaj/Amaj
        song = transpose(song)
        
        # encode song
        encoded_song = encode_song(song)
        
        # save song to text file
        save_path = os.path.join(SAVE_DIR,str(i))
        with open(save_path,'w') as f:
            f.write(encoded_song)

    
    songs[0].show()
    



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

def transpose(song):
    # get key from song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
    
    print(key)

    # get interval from transposition. Eg , Bmaj -> Cmaj
    
    if key.mode == 'major':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == 'minor':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    
    # transpose song by calculated interval
    transposed_song = song.transpose(interval)
    
    return transposed_song

def load(file_path):
    with open(file_path,'r') as f:
        song  = f.read()
    return song

def create_single_file_dataset(dataset_path,file_dataset_path,sequence_length):
    delimeter = "/ " * sequence_length 
    
    songs = ""
    for path,_,files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path,file)
            song = load(file_path)
            songs = songs + song + " " + delimeter
            
    songs = songs[:-1]
    
    # save string that contains all the dataset
    with open(file_dataset_path,"w") as f:
        f.write(songs)
        
    return songs

# songs = create_single_file_dataset(SAVE_DIR,SINGLE_FILE,SEQUENCE_LENGTH)

def create_mapping(songs,mapping_path):
    mappings = {}

    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))
    
    # creating mappings
    for i , symbol in enumerate(vocabulary):
        mappings[symbol] = i 

    # save vocabulary to  a json file

    with open(mapping_path,'w') as f:
     json.dump(mappings,f,indent=4) 

inputs = ""
targets = ""
def main():
    preprocess(KERN_DATASET_PATH)
    global songs
    songs = create_single_file_dataset(SAVE_DIR,SINGLE_FILE,SEQUENCE_LENGTH)
    create_mapping(songs,MAPPING_PATH)
    global inputs,targets
    inputs,targets = generate_training_sequences(SEQUENCE_LENGTH)
    
    a = 1

# main()

# len(songs)

def convert_songs_to_int(songs):

    int_songs = []
    # load the mappings 
    mappings = json.load(open("mapping.json",'r'))

    # cast songs strings to a list
    
    songs = songs.split()

    # map songs to list 
    for symbol in songs:
        int_songs.append(mappings[symbol])
        
    return int_songs

def generate_training_sequences(sequence_length):
    # [ 11, 12, 13, 14]
    
    # load songs and map them to int
    songs = load(SINGLE_FILE)
    int_songs = convert_songs_to_int(songs)

    # generate the training sequence
    
    inputs = []
    targets = []

    num_sequences = (len(int_songs)) - sequence_length 
    
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # one hot encoding
    
    vocabulary_size = len(set(int_songs))
    
    inputs = keras.utils.to_categorical(inputs,num_classes = vocabulary_size)
    
    targets = np.array(targets)
    
    return inputs,targets



