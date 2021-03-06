
import pandas as pd
from tqdm import tqdm
from tools import *
import argparse

parser = argparse.ArgumentParser(description='Data processing for ChordLSTM Model')
parser.add_argument('--data', type=str, default='wikifonia-db',
                    help='data folder to process: wikifonia-db/jazz')
args = parser.parse_args()


######################################
#                                    #
#        EXTRACTION OF DATA          #
#                                    #
######################################

if args.data == 'jazz':
    save_prefix = 'jazz_'
else:
    save_prefix = ''

FILES_PATH = 'wikifonia-db/'
# path name: path for directory of file names (not db files) - only useful for jazz mode -
path_name = args.data + '/'
name_list = make_list_of_names(path_name)


def xml_to_db(name_list):
    list_ = []
    for file_name in name_list:
        name = file_name.split('.mx')[0]
        list_.append(name + '.db')
    return list_


def jazz_files(list_):
    ref_list = make_list_of_names('wikifonia-db/')
    jazz_list = xml_to_db(name_list)
    list_ = []
    for file_name in jazz_list:
        if file_name in ref_list:
            list_.append(file_name)
        else:
            print(file_name, ' have been lost in conversion')
    return list_


if args.data == 'jazz':
    name_list = jazz_files(name_list)


def make_dictionary(name_list):
    data = {}
    for file_name in tqdm(name_list):
        df = df = pd.read_csv(FILES_PATH + file_name, skiprows=1, names=["pos_bar", "dur", "note", "chord"])
        i = 0
        while i < len(df):
            non_rest = 0
            bar = 0
            melody = []
            note = df.iloc[i, 2]
            if note != 'rest':
                non_rest += 1
            dur = str(df.iloc[i, 1])
            chord = df.iloc[i, -1]
            dur_has_tri = len(dur.split('/')) > 1 and int(dur.split('/')[-1]) % 3 == 0
            dur_has_quint = len(dur.split('/')) > 1 and int(dur.split('/')[-1]) % 5 == 0
            if dur != '0':
                melody = [(note, dur)]
            j = 1
            while i + j < len(df) and bar < 2 and not ((df.iloc[i + j, 0] == '0' or df.iloc[i + j, 0] == 0) and bar == 1) and df.iloc[i + j, -1] == chord:
                note = df.iloc[i + j, 2]
                dur = str(df.iloc[i + j, 1])
                if df.iloc[i + j, 0] == '0' or df.iloc[i + j, 0] == 0:
                    bar += 1
                if dur != '0':
                    melody.append((note, dur))
                if note != 'rest':
                    non_rest += 1
                if len(dur.split('/')) > 1 and int(dur.split('/')[-1]) % 3 == 0:
                    dur_has_tri = True
                if len(dur.split('/')) > 1 and int(dur.split('/')[-1]) % 5 == 0:
                    dur_has_quint = True
                j += 1
            if (non_rest >= 2) and (not dur_has_tri) and (not dur_has_quint):
                if chord in data:
                    data[chord].append(melody)
                else:
                    data[chord] = [melody]
            i += j
    data.pop('NC', None)
    return data


start = time.time()

print('-' * 50)
print('Creating dataset')
print('-' * 50)

data = make_dictionary(name_list)

# Dump the dataset into "data.pkl"
with open('data/' + save_prefix + 'data.pkl', 'wb') as f:
    pickle.dump(data, f)

print('Elapsed time extracting data: ', time_since(start))

######################################
#                                    #
#        RESCALING & PADDING         #
#                                    #
######################################

def get_duration_scale(data):
    list_denominator = []
    dur_list = duration_list(data)
    tmp = [get_denominator(str(c)) for c in dur_list]
    list_denominator.extend(tmp)
    a = lcm(int(list_denominator[0]), int(list_denominator[1]))
    for i in range(3, len(list_denominator)):
        if int(list_denominator[i]) > 1:
            a = lcm(int(a), int(list_denominator[i]))
    return a


def duration_list(dictionary):
    list_ = []
    for chord in dictionary:
        for sequence in dictionary[chord]:
            for x in sequence:
                list_.append(x[1])
    return list(set(list_))


def duration_rescale(dictionary, scaling_factor):
    new_dictionary = {}
    for chord in dictionary:
        for sequence in dictionary[chord]:
            l = []
            for x in sequence:
                l.append((note_to_midi(x[0]), fraction_to_int(x[1]) * scaling_factor))
            if chord in new_dictionary:
                new_dictionary[chord].append(l)
            else:
                new_dictionary[chord] = [l]
    return new_dictionary


def duration_spacing(dataset):
    new_dictionary = {}
    for chord in dataset:
        for sequence in dataset[chord]:
            l = []
            for note, dur in sequence:
                l.append(note)
                l.extend(['hold'] * (int(dur) - 1))
            if chord in new_dictionary:
                new_dictionary[chord].append(l)
            else:
                new_dictionary[chord] = [l]
    return new_dictionary


def pad_data(data, max_length):
    new_data = {}
    for chord in data:
        for melody in data[chord]:
            if len(melody) < max_length:
                melody.extend(['PAD'] * (max_length - len(melody)))
            if chord in new_data:
                new_data[chord].append(melody)
            else:
                new_data[chord] = [melody]
    return new_data


def dataset_quantization(data):
    dataset = {}
    for chord in data:
        for melody in data[chord]:
            l = []
            for x in melody:
                if x[0] != 'rest':
                    note = x[0] % 12
                    l.append((note, x[1]))
                else:
                    l.append((x[0], x[1]))
            if chord in dataset:
                dataset[chord].append(l)
            else:
                dataset[chord] = [l]
    dataset = duration_spacing(dataset)
    return dataset


start = time.time()

scaling_factor = get_duration_scale(data)
print('Scaling factor: ', scaling_factor)
data = duration_rescale(data, scaling_factor)

data_quantized = dataset_quantization(data)

# Dump the spaced dataset
with open('data/' + save_prefix + 'data_rescaled.pkl', 'wb') as f:
    pickle.dump(data_quantized, f)

print('Elapsed time for quantization: ', time_since(start))


######################################
#                                    #
#           TRANSPOSITION            #
#                                    #
######################################

#import pickle
#from music21.note import Note
#from music21.interval import Interval
#from ls import spotify_chord as sc
#from tqdm import tqdm
#from functools import lru_cache


#def melody_transpose(melody, transposition):
#    transposed_melody = []
#    for note in melody:
#        if note != 'rest' and note != 'hold' and note != 'PAD':
#            transposed_melody.append((note + transposition) % 12)
#        else:
#            transposed_melody.append(note)
#    return transposed_melody


#@lru_cache(maxsize=10000)
#def chord_transpose(chord, transposition):
#    scs = sc.SpotifyChord.from_figure(chord)
#    new_scs = scs.transposed_by(Interval(transposition))
#    return new_scs
#

#def data_transpose(chord, melody, transposition):
#    return chord_transpose(chord, transposition), melody_transpose(melody, transposition)
#

#def data_augmentation(data):
#    new_data = {}
#    for chord in tqdm(data):
#        for melody in data[chord]:
#            for i in range(13):
#                if i == 0:
#                    new_chord = sc.SpotifyChord.from_figure(chord)
#                    if new_chord in new_data:
#                        new_data[new_chord].append(melody)
#                    else:
#                        new_data[new_chord] = [melody]
#                else:
#                    new_chord, new_melody = data_transpose(chord, melody, i)
#                    if new_chord in new_data:
#                        new_data[new_chord].append(new_melody)
#                    else:
#                        new_data[new_chord] = [new_melody]
#    return new_data


#start = time.time()

#print('-' * 50)
#print('Transposing dataset')
#print('-' * 50)

# Data augmentation
#data_augmented = data_augmentation(data)
#data_augmented = data_augmentation(data_quantized)

# Dump rescaled and augmented dataset into "data_augmented.pkl"
#with open('data/test/' + save_prefix + 'data_augmented.pkl', 'wb') as f:
#    pickle.dump(data_augmented, f)

#print('Elapsed time for transposition: ', time_since(start))


######################################
#                                    #
#        CHORD REPRESENTATION        #
#                                    #
######################################

def pitch_list(chord):
    pitch_list = [p.midi for p in chord.chord.pitches]
    pitch_list.sort()
    # Keep only 4 1st pitches
    short_pitch_list = [p % 12 for p in pitch_list]
    return short_pitch_list


def convert_dictionary(data):
    print('-' * 50)
    print('Converting Dictionary')
    print('-' * 50)
    new_data = {}
    for chord in tqdm(data):
        sc_chord = sc.SpotifyChord.from_figure(chord)
        new_data[sc_chord] = data[chord]
    return new_data


start = time.time()
# Changing chord representation from string to spotify chord
data = convert_dictionary(data_quantized)


# Dump rescaled and augmented dataset into "data_augmented.pkl"
with open('data/' + save_prefix + 'data_converted.pkl', 'wb') as f:
    pickle.dump(data, f)

print('Elapsed time for moving to SpotifyChord representation: ', time_since(start))


######################################
#                                    #
#        DEQUE REPRESENTATION        #
#                                    #
######################################

start = time.time()

data = dictitonary_to_deque(data)
# Dump rescaled and augmented dataset into "data_augmented.pkl"
with open('data/' + save_prefix + 'dataset.pkl', 'wb') as f:
    pickle.dump(data, f)

print('Elapsed time for moving to deque representation: ', time_since(start))

print('-' * 50)
print('End of data processing')
print('-' * 50)
