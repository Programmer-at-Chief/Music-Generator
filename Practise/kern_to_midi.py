from music21 import converter

# Load the KRN file
krn_file = 'deut0567.krn'

# Parse the KRN file
score = converter.parse(krn_file)

# Export the score to a MIDI file
midi_file = "output_file.mid"
score.write("midi", fp=midi_file)

print(f"Conversion complete! MIDI file saved as {midi_file}")
