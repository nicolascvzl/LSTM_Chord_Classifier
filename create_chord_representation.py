from tools import *
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pitch_list(chord):
    pitch_list = [p.midi for p in chord.chord.pitches]
    pitch_list.sort()
    # Keep only 4 1st pitches
    short_pitch_list = [p % 12 for p in pitch_list]
    return short_pitch_list


@lru_cache(maxsize=10000)
def sc_to_tensor(chord):
    chord_pitch = pitch_list(chord)
    chord_tensor = torch.zeros(12).to(device)
    for p in chord_pitch:
        chord_tensor[p] = 1
    return chord_tensor


def tensor_in_keys(tensor, dictionary):
    keys = list(dictionary.keys())
    for k in keys:
        if (tensor == k).sum().item() == len(tensor):
            return True, k
    return False, torch.Tensor([0])


with open('data/dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

chords = list(set([sequence[1] for sequence in dataset]))
augmented_chords = []
for chord in chords:
    if chord not in augmented_chords:
        augmented_chords.append(chord)
    for i in range(1, 12):
        transposed_chord = chord_transpose(chord, transposition=i)
        if transposed_chord not in augmented_chords:
            augmented_chords.append(transposed_chord)
augmented_chords = list(set(augmented_chords))

chord_representation = {}
print('Creating spotify_chord/chord_tensor look-up table...')
for spotify_chord in tqdm(augmented_chords):
    chord_tensor = sc_to_tensor(spotify_chord)
    b, k = tensor_in_keys(chord_tensor, chord_representation)
    if b:
        chord_representation[k].append(spotify_chord)
    else:
        chord_representation[chord_tensor] = [spotify_chord]
print('Done.')

print('Saving representation...')
with open('data/chord_representation.pkl', 'wb') as f:
    pickle.dump(chord_representation, f)
print('Done.')