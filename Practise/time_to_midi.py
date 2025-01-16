import music21 as m21

encoded_song = "69 _ _ _ 72 _ _ _ 72 _"

def decode_song(encoded_song, time_step=0.25):
    decoded_song = m21.stream.Score()
    current_event = []
    symbols = encoded_song.split(" ")

    for symbol in symbols:
        # If symbol is a note (i.e., an integer, representing MIDI pitch)
        if symbol != "_":
            if current_event:  # If there was an ongoing note/rest
                # Check previous note/rest and close it by adding to the song
                note_or_rest = current_event[0]
                duration = len(current_event) * time_step
                if isinstance(note_or_rest, m21.note.Note):
                    note_or_rest.duration = m21.duration.Duration(duration)
                    decoded_song.append(note_or_rest)
                elif isinstance(note_or_rest, m21.note.Rest):
                    note_or_rest.duration = m21.duration.Duration(duration)
                    decoded_song.append(note_or_rest)

            # Start a new note/rest
            if symbol == 'r':
                current_event = [m21.note.Rest()]
            else:
                current_event = [m21.note.Note(int(symbol))]

        else:  # If it's an underscore ("_"), just continue building the current event
            current_event.append(current_event[0] if current_event else None)

    # Finalize the last note/rest
    if current_event:
        note_or_rest = current_event[0]
        duration = len(current_event) * time_step
        if isinstance(note_or_rest, m21.note.Note):
            note_or_rest.duration = m21.duration.Duration(duration)
            decoded_song.append(note_or_rest)
        elif isinstance(note_or_rest, m21.note.Rest):
            note_or_rest.duration = m21.duration.Duration(duration)
            decoded_song.append(note_or_rest)

    return decoded_song

# Example usage:
decoded_song = decode_song(encoded_song)

# Show the reconstructed song (as a string or visually)
decoded_song.show('text')  # or decoded_song.show() to visualize it

def save_song_as_midi(decoded_song, output_file_name="output_file.mid"):
    # Create a MIDI file from the decoded music21 stream (Score)
    midi_stream = m21.midi.translate.music21ObjectToMidiFile(decoded_song)

    # Save the MIDI file to disk
    midi_stream.open(output_file_name, 'wb')
    midi_stream.write()
    midi_stream.close()

# Example usage:
# Assuming 'decoded_song' is the music21.stream.Score object you decoded from the encoded song
save_song_as_midi(decoded_song, "reconstructed_song.mid")
