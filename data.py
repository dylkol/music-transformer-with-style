import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
from midi2audio import FluidSynth
from os import path, listdir
from typing import Tuple


def encode_midi(midi: MidiFile, quantization=16) -> Tuple[np.array, np.array, np.array, np.array, MidiTrack]:
    """
    Encode midi as a piano roll representation merging all tracks.
    :param midi: ``MidiFile`` as loaded by Mido library.
    :param quantization: Number of timesteps for each beat (independent of time signature).
    :return: A tuple containing:
        1. a binary matrix for the key being on (1) or off (0), shape (note, time)
        2. a dynamics matrix (cont. values 0-1), shape (note, time)
        3. a repetition matrix for repeating notes, 1 if repeated, 0 if not, shape (note, time)
        4. a matrix denoting the note's position in a bar as a one-hot encoded value, shape (time, pos)
        5. a track with containing only the MIDI's metadata.
    """

    tpb = midi.ticks_per_beat
    npb = 1 / 4
    tpn = tpb / npb
    ticks_per_bar = tpn  # Default is one full note per bar (4/4)
    tick_quantile = ticks_per_bar / quantization

    pressed = np.zeros((88, 100 * quantization), dtype=np.int8)
    dynamics = np.zeros((88, 100 * quantization), dtype=np.float32)
    repeated = np.zeros((88, 100 * quantization), dtype=np.int8)
    meta_track = MidiTrack()
    for track in midi.tracks:
        curr_time = 0
        last_released = np.full(88, -1, dtype=np.int32)  # To track repeated notes
        last_pressed = np.full((88, 2), (-1, -1), dtype=np.int32)  # To track duration and dynamics
        for msg in track:
            if msg.is_meta:
                meta_track.append(msg)
            if msg.type == 'time_signature':
                npb = msg.notated_32nd_notes_per_beat / 32
                tpn = tpb / npb
                ticks_per_bar = (msg.numerator / msg.denominator) * tpn
                tick_quantile = ticks_per_bar / quantization

            if msg.type == "note_on" and msg.velocity > 0:
                # Note pressed
                curr_note = np.clip(msg.note, 21, 108) - 21  # Clip to piano range and reindex from 0-88
                new_time = curr_time + int(np.round(msg.time / tick_quantile))
                if new_time >= pressed.shape[1]:
                    # Increase time size of array in case if the current note crosses the threshold
                    pressed = np.pad(pressed, [(0, 0), (0, int(0.2 * new_time))])
                    dynamics = np.pad(dynamics, [(0, 0), (0, int(0.2 * new_time))])
                    repeated = np.pad(repeated, [(0, 0), (0, int(0.2 * new_time))])
                if new_time == last_released[curr_note]:
                    # Note repeated
                    repeated[curr_note, new_time] = 1
                last_pressed[curr_note] = [new_time, msg.velocity]
                curr_time = new_time
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                # Note released
                curr_note = np.clip(msg.note, 21, 108) - 21  # Clip to piano range and reindex from 0-88
                new_time = curr_time + int(np.round(msg.time / tick_quantile))
                last_released[curr_note] = new_time
                if last_pressed[curr_note, 0] == -1:
                    raise Exception(f"Error in message: {msg}. Note has not been played yet")
                if new_time >= pressed.shape[1]:
                    # Increase time size of array in case if the current note crosses the threshold
                    pressed = np.pad(pressed, [(0, 0), (0, int(0.2 * new_time))])
                    dynamics = np.pad(dynamics, [(0, 0), (0, int(0.2 * new_time))])
                    repeated = np.pad(repeated, [(0, 0), (0, int(0.2 * new_time))])
                pressed[curr_note, last_pressed[curr_note, 0]:new_time] = 1
                dynamics[curr_note, last_pressed[curr_note, 0]:new_time] = last_pressed[curr_note, 1] / 127
                curr_time = new_time
            else:
                curr_time += int(np.round(msg.time / tick_quantile))
    pressed = pressed[:, :new_time + 1]
    dynamics = dynamics[:, :new_time + 1]
    repeated = repeated[:, :new_time + 1]

    beats = np.tile(np.arange(quantization), int(np.ceil(dynamics.shape[1] / quantization)))[:new_time + 1]
    beats_onehot = np.identity(quantization)[beats]
    return pressed, dynamics, repeated, beats_onehot, meta_track


def decode_midi(dynamics, repeated, meta_track: MidiTrack = None, quantization=16, bpm=120):
    """
    Decodes the piano roll representation as encoded by ``encode_midi`` to a Mido ``MidiFile``.
    :param dynamics: Dynamics matrix (cont. values 0-1), shape (note, time).
    :param repeated: Repetition matrix for repeating notes, 1 if repeated, 0 if not, shape (note, time)
    :param meta_track: Optional meta track to add to the MIDI, if ``None`` (default) 4/4 time is used throughout.
    :param quantization: Number of notes in a bar.
    :param bpm: If ``meta_track`` is ``None``, sets bpm to this value for 4/4 time.
    :return: Mido ``MidiFile`` containing the decoded representation in a single track (apart from the meta track).
    """
    mid = MidiFile(type=1)
    track = MidiTrack()
    track.append(Message('program_change', channel=0, program=0, time=0))  # Specify piano as instrument
    if meta_track is None:
        meta_track = MidiTrack()
        meta_track.append(MetaMessage('time_signature', numerator=4, denominator=4,
                                      notated_32nd_notes_per_beat=8, time=0))

    mid.tracks.append(track)

    tpb = mid.ticks_per_beat
    npb = 1 / 4
    tpn = tpb / npb
    ticks_per_bar = tpn  # Default is one full note per bar (4/4)
    tick_quantile = ticks_per_bar / quantization
    tick_quantiles = []
    curr_time = 0
    curr_time_signature = (4,4)
    for idx, msg in enumerate(meta_track):
        if msg.type == 'time_signature':
            curr_time += int(np.round(msg.time / tick_quantile))
            npb = msg.notated_32nd_notes_per_beat / 32
            tpn = tpb / npb
            ticks_per_bar = (msg.numerator / msg.denominator) * tpn
            tick_quantile = ticks_per_bar / quantization
            tick_quantiles.append((curr_time, tick_quantile))
            curr_time_signature = (msg.numerator, msg.denominator)
    # Add tempo at the end, denoting tempo for generated part
    meta_track.append(MetaMessage('set_tempo', tempo=bpm2tempo(bpm, curr_time_signature), time=0))
    curr_pressed = np.full(88, False)
    last_msg_time = 0
    curr_tick_quantile_idx = 0
    for t in range(dynamics.shape[1]):
        if len(tick_quantiles) > curr_tick_quantile_idx + 1 and tick_quantiles[curr_tick_quantile_idx + 1][0] >= curr_time:
            curr_tick_quantile_idx += 1
        curr_quantile = int(tick_quantiles[curr_tick_quantile_idx][1])
        for n in range(88):
            if repeated[n, t] == 1:
                # Repeated note detected, add off and on message
                track.append(Message("note_on", note=n + 21, velocity=0, time=(t - last_msg_time) * curr_quantile))
                track.append(Message("note_on", note=n + 21, velocity=int(dynamics[n, t] * 127), time=0))
                last_msg_time = t
            elif dynamics[n, t] == 0 and curr_pressed[n] > 0:
                # Note off, send off message and reset time
                track.append(Message("note_on", note=n + 21, velocity=0, time=(t - last_msg_time) * curr_quantile))
                curr_pressed[n] = False
                last_msg_time = t
            elif dynamics[n, t] > 0 and curr_pressed[n] == 0:
                # Note newly on, add message
                track.append(Message("note_on", note=n + 21, velocity=int(dynamics[n, t] * 127),
                                     time=(t - last_msg_time) * curr_quantile))
                curr_pressed[n] = True
                last_msg_time = t
    mid.tracks.append(meta_track)
    return mid


def midi_to_wav(midi_file: str, soundfont=None):
    """
    Convert MIDI file to WAV file for playback in browser.
    :param midi_file: Path to the MIDI file.
    :param soundfont: Optional soundfont to use in conversion.
    :return: Path to new WAV file.
    """
    out_path = midi_file.split('.mid')[0] + '.wav'
    if soundfont is not None:
        fs = FluidSynth(soundfont)
    else:
        fs = FluidSynth()
    fs.midi_to_audio(midi_file, out_path)
    return out_path


class PianoDataset(Dataset):
    """
    Dataset for the piano-midi data.
    """
    def __init__(self, composers: list[str], quantization: int = 16, transposition: int = 3):
        """
        Create dataset for the list of composers.
        :param composers: List of composers to create dataset for, each piece must be stored in a folder with the same name as the composer.
        :param quantization: Number of timesteps in a bar, independent of time signature.
        :param transposition: Transposed variants ``-transposition`` to ``transposition`` are added to the dataset. 0 for no added transposition.
        """
        self.pieces = []
        id_matrix = np.eye(len(composers))
        # (piece, time, note)
        for idx, composer in enumerate(composers):
            style = id_matrix[idx]
            for file in listdir(path.join("data", composer)):
                pressed, dynamics, repeated, beats, _ = encode_midi(MidiFile(path.join("data", composer, file)),
                                                                    quantization=quantization)
                self.pieces.append((pressed, dynamics, repeated, beats, style))
                for t in range(-transposition, transposition):
                    # Transpose to other keys
                    self.pieces.append((np.roll(pressed, t, 0), np.roll(dynamics, t, 0), np.roll(repeated, t, 0), beats, style))

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        return self.pieces[idx]


def load_data(composers, quantization=16, transposition=3, seed=None, shuffle=True) -> Tuple[DataLoader, DataLoader]:
    """
    Loads composer data in train and valid dataloaders.
    :param composers: List of composers to generate dataset for.
    :param quantization: Number of timesteps in a bar, independent of time signature.
    :param transposition: Transposed variants ``-transposition`` to ``transposition`` are added to the dataset. 0 for no added transposition.
    :param seed: Seed for shuffling, default is None for random seed.
    :param shuffle: Whether to shuffle data (default: True)
    :return: Train dataloader (90%) and validation dataloader (10%)
    """
    ds = PianoDataset(composers, quantization, transposition)
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None
    train, valid = random_split(ds, [0.9, 0.1], generator=gen)
    train_dl = DataLoader(train, batch_size=1, shuffle=shuffle, generator=gen)
    valid_dl = DataLoader(valid, batch_size=1, shuffle=shuffle, generator=gen)
    return train_dl, valid_dl


