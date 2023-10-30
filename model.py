from os import path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mido import MidiTrack
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from typing import Callable, Tuple

from data import decode_midi


def skew(q_e: torch.Tensor) -> torch.Tensor:
    """
    "Skewing procedure for  as described in Huang et al., section 3.4"
    :param q_e: QE^T, LxL query matrix multiplied with embeddings for relative positions.
    :return: LxL matrix denoting relative importance values indexed by (query, key) positions.
    """
    # Skewing procedure
    s_rel = F.pad(q_e, (1, 0))  # Pad one col on left (batch, head, seq, seq+1)
    s_rel = s_rel.reshape(*s_rel.shape[:2], s_rel.shape[3], s_rel.shape[2])  # (batch, head, seq+1, seq)
    s_rel = s_rel[:, :, 1:, :]  # (batch, head, seq, seq)
    return s_rel


def add_positional_encoding(input: torch.Tensor):
    """
    Add positional encoding from regular transformer (Attention is all you Need). Used as an extra on top of
    relative position matrix. Modifies inplace.
    :param input: Input of shape (batch, seq, feature)
    """
    position_div = 10000 ** (2 * torch.arange(0, input.shape[2], 2) / input.shape[2]).to(input.device)
    positions = torch.arange(input.shape[1]).view(1, input.shape[1], 1).to(input.device)
    input[:, :, 0::2] += torch.sin(positions / position_div)
    input[:, :, 1::2] += torch.cos(positions / position_div)


class RelativeAttentionDecoderLayer(nn.Module):
    """
    Decoder layer for a relative attention transformer
    """

    def __init__(self, d_model, nhead, max_dist, dim_fc, dropout=0.1):
        """
        :param d_model: Feature dimensionality.
        :param nhead: Number of heads.
        :param max_dist: Maximum distance for the relative positions, i.e. longest sequence that can go into the layer
        :param dim_fc: Dimensions of hidden FC layer.
        :param dropout: Dropout rate in FC layer.
        """
        super().__init__()
        if d_model % nhead != 0:
            raise Exception("Number of heads must be divisible by input feature dims")

        self.d_model = d_model
        self.n_heads = nhead

        self.max_dist = max_dist
        self.wk = nn.Linear(d_model, d_model)
        self.wq = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.rel_embedding = nn.Embedding(max_dist, d_model)
        self.fc = nn.Sequential(nn.Linear(d_model, dim_fc),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(dim_fc, d_model))

    def forward(self, input: torch.Tensor, mask=None) -> torch.Tensor:
        """
        :param input: Input of shape (batch, seq, feature)
        :param mask: Mask to apply to input (prevents looking forward)
        :return: Attention values of shape (batch, seq, feature)
        """
        # Input:
        k = self.wk(input).view(*input.shape[:2], self.n_heads,
                                self.d_model // self.n_heads)  # (batch, seq, head, feature)
        q = self.wq(input).view(*input.shape[:2], self.n_heads, self.d_model // self.n_heads)
        v = self.wv(input).view(*input.shape[:2], self.n_heads, self.d_model // self.n_heads)

        # Rel embeddings go from max_dist - seq len up to max dist so that non-used distances correspond to the first embeddings, as they go from largest distance to smallest
        embeddings = self.rel_embedding(torch.arange(self.max_dist - input.shape[1], self.max_dist, dtype=torch.int).to(
            input.device))  # (seq, feature)
        embeddings = embeddings.view(embeddings.shape[0], self.n_heads,
                                     self.d_model // self.n_heads)  # (seq, head, feature)

        # (batch, head, seq, feature) * (head, feature, seq) -> (batch, head, seq, seq)
        q_e = torch.matmul(q.transpose(1, 2), embeddings.permute(1, 2, 0))
        s_rel = skew(q_e)

        # (batch, head, seq, feature) * (batch, head, feature, seq) -> (batch, head, seq, seq)
        qk = torch.matmul(q.transpose(1, 2), k.permute(0, 2, 3, 1))

        # Get the final attention values using qk and s_rel
        logits = (qk + s_rel) / torch.sqrt(torch.tensor(self.d_model / self.n_heads).to(input.device))
        if mask is not None:
            logits += mask
        softmax_out = F.softmax(logits, -1)  # (batch, head, seq)

        # (batch, head, seq) * (batch, head, seq, feature) -> (batch, head, seq, feature)
        attention = torch.matmul(softmax_out, v.transpose(1, 2))
        # concat heads
        attention = attention.transpose(1, 2).reshape(attention.shape[0], attention.shape[2], self.d_model)
        # final pass through FC network
        return self.fc(attention)  # (batch, seq, d_model)


class MusicStyleRelativeTransformer(nn.Module):
    """
    Music generating model using a Relative Transformer layers, conditioned on composer style.
    """

    def __init__(self, d_note: int, d_style: int, tf_layers: int, n_heads: int, max_len: int,
                 d_fc: int, num_composers: int, p_dropout=0.2, bar_quantization=16):
        """
        :param d_note: Note feature dimensionality after passing through "note octave" layer.
        :param d_style: Dimensionality of composer style embeddings.
        :param tf_layers: Number of relative transformer layers.
        :param n_heads: Number of heads in each transformer layer.
        :param max_len: Maximum sequence length able to go into transformer layers.
        :param d_fc: Dimensionality of hidden layer in FC network in transformer layers.
        :param num_composers: Number of composers used for style embedding.
        :param p_dropout: Dropout proportion in various layers.
        :param bar_quantization: Number of timesteps in a bar (independent of time signature).
        """
        super().__init__()
        self.note_octaves = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Conv1d(1, d_note, 23, padding=11),
            nn.ReLU())
        self.style_layer = nn.Linear(num_composers, d_style)
        self.bar_quantization = bar_quantization

        self.style_time = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(d_style, d_note + bar_quantization),
            nn.ReLU())
        # self.style_note = nn.Sequential(
        #     nn.Dropout(p_dropout),
        #     nn.Linear(d_style, d_note + bar_quantization),
        #     nn.ReLU())
        self.time_transformer = nn.ModuleList([
            RelativeAttentionDecoderLayer(d_model=d_note + bar_quantization,
                                          nhead=n_heads,
                                          max_dist=max_len,
                                          dim_fc=d_fc,
                                          dropout=p_dropout) for i in range(tf_layers)])
        # self.note_transformer = nn.ModuleList([
        #     RelativeAttentionDecoderLayer(d_model=d_note + bar_quantization,
        #                                   nhead=n_heads//2,
        #                                   max_dist=88,
        #                                   dim_feedforward=d_feedforward,
        #                                   dropout=p_dropout) for i in range(tf_layers)])
        self.play_dense = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(d_note + bar_quantization, 2),  # Play probability and replay probability
        )  # No sigmoid so we can include temperature in forward
        self.dynamics_dense = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(d_note + bar_quantization, 1),  # Dynamics of note
            nn.Sigmoid())

    def forward(self, pressed: torch.Tensor, beats: torch.Tensor, style: torch.Tensor, temp=1.0) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        :param pressed: Piano roll representation of shape (note, time) 1 denoting a note pressed, 0 unpressed.
        :param beats: Beats as one-hot encoded numbers in (0, bar_quantization], in shape (time, bar_quantization)
        :param style: Style as a one-hot encoded number denoting the composer, shape (num_composers)
        :param temp: Temperature factor, final output before sigmoid is divided by this factor. Higher means more constant, but less creative output.
        :return: Tuple containing:
        1. Tensor ``played``, of shape (note, time, 2), denoting play and replay probability of notes.
        2. Tensor ``dynamics`` of shape (note, time), denoting volume on a scale 0-1 of notes.
        """
        notes_reshaped = pressed.unsqueeze(1).transpose(0, 2)  # (time, channel, note)
        beats_reshaped = beats.repeat((88, 1, 1)).permute(1, 0, 2)  # (time, note, beat_num)

        note_features = self.note_octaves(notes_reshaped).transpose(1, 2)  # (time, note, note feature)
        notes_with_beats = torch.cat([note_features, beats_reshaped], dim=-1)  # add beats as features for each note
        style_embedding = self.style_layer(style)  # (embedding_dims)
        time_mask = nn.Transformer.generate_square_subsequent_mask(notes_with_beats.shape[0],
                                                                   device=notes_with_beats.device)

        out = notes_with_beats.transpose(0, 1)  # (note, time, note feature)
        add_positional_encoding(out)
        for tf_layer in self.time_transformer:
            out += self.style_time(style_embedding)
            out = tf_layer(out, mask=time_mask)
        out = out.transpose(0, 1)

        # In DeepJ, there's a 'note axis' LSTM after the 'time axis', but adding a transformer for this makes the output much worse
        # note_mask = nn.Transformer.generate_square_subsequent_mask(notes_with_beats.shape[1], device=notes_with_beats.device)
        # out = out.transpose(0, 1)  # (time, note, note feature)
        # add_positional_encoding(out)
        # for tf_layer in self.note_transformer:
        # out += self.style_note(style_embedding)
        # out = tf_layer(out, mask=note_mask)

        played = torch.zeros(88, out.shape[0], 2).to(out.device)
        dynamics = torch.zeros(88, out.shape[0]).to(out.device)

        for t in range(out.shape[0]):
            play_dense_out = self.play_dense(out[t])
            play_dense_out[:, 0] /= temp
            played[:, t, :] = play_dense_out.sigmoid()
            dynamics[:, t] = self.dynamics_dense(out[t])[0]

        return played, dynamics


def deepj_loss(pred_played: torch.Tensor, pred_dynamics: torch.Tensor, true_pressed: torch.Tensor,
               true_repeated: torch.Tensor, true_dynamics: torch.Tensor) -> torch.Tensor:
    """
    Calculates total loss as defined in the DeepJ paper as a sum of
    CE loss of played notes, CE loss of repeated notes, and MSE loss of dynamics.
    :param pred_played: Output play and replay probabilities of transformer model, shape (note, time, 2).
    :param pred_dynamics: Output dynamics value of transformer model, shape (note, time).
    :param true_pressed: True values of notes pressed (1) or unpressed (0), shape (note, time).
    :param true_repeated: True values of whether notes are repeated (1) or not (0), shape (note, time).
    :param true_dynamics: True values of note volume as values from 0 to 1, shape (note, time).
    :return: Loss value in a zero-dim tensor.
    """
    # Cross entropy input should be a vector [p, 1-p]
    l_played = nn.functional.binary_cross_entropy(pred_played[:, :, 0], true_pressed)
    l_repeated = nn.functional.binary_cross_entropy((true_pressed * pred_played[:, :, 1]),
                                                    true_repeated)  # Repeats only counted if note is played
    l_dynamics = nn.functional.mse_loss((true_pressed * pred_dynamics),
                                        true_dynamics)  # Dynamics zero if note isn't played
    return l_played + l_repeated + l_dynamics


def train_model(model: MusicStyleRelativeTransformer, train: DataLoader, valid: DataLoader,
                device: torch.device, lr=1e-3, epochs=3, bs=128, overlap=0, filename="transformer.pt"):
    """
    Trains a ``MusicStyleRelativeTransformer model``.
    :param model: The model to train.
    :param train: Dataloader of training data.
    :param valid: Dataloader of validation data, evaluated after every epoch.
    :param device: Device to perform training on.
    :param lr: Learning rate of Adam optimizer.
    :param epochs: Number of epochs.
    :param bs: Batch size in each piece (number of timesteps).
    :param overlap: Number of timesteps that consecutive batches overlap.
    :param filename: Filename to save trained model to in ``model`` directory.
    """

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if not filename.endswith(".pt"):
        filename = filename + ".pt"
    if overlap >= bs:
        raise Exception("Overlap cannot be bigger or equal to batch size")
    for epoch in range(epochs):
        model.train()
        print("=" * 50 + f"Epoch {epoch}" + "=" * 50)
        total_loss = 0
        total_batches = 0
        for piece_num, (pressed, dynamics, repeated, beats, style) in enumerate((tqdm(train))):
            pressed = pressed.squeeze(0).to(torch.float32)
            dynamics = dynamics.squeeze(0).to(torch.float32)
            repeated = repeated.squeeze(0).to(torch.float32)
            beats = beats.squeeze(0).to(torch.float32)
            style = style.squeeze(0).to(torch.float32)
            piece_batches = np.ceil(
                (pressed.shape[1] - 1 - bs) / (bs - overlap)) + 1  # Conv output formula with overlap = bs-stride
            start = 0
            end = min(bs, pressed.shape[1] - 1)
            for batch_num in range(int(piece_batches)):
                batch_pressed = pressed[:, start:end].to(device)
                batch_dynamics = dynamics[:, start:end].to(device)
                batch_repeated = repeated[:, start:end].to(device)
                batch_beats = beats[start:end].to(device)

                opt.zero_grad()
                pred_played, pred_dynamics = model(batch_pressed, batch_beats,
                                                   style.to(device))
                pred_played = pred_played[:, overlap:]  # Only consider predictions that were not part of prev batch
                pred_dynamics = pred_dynamics[:, overlap:]

                true_pressed = pressed[:, start + 1 + overlap:end + 1].to(device)
                true_repeated = repeated[:, start + 1 + overlap:end + 1].to(device)
                true_dynamics = dynamics[:, start + 1 + overlap:end + 1].to(device)

                loss = deepj_loss(pred_played, pred_dynamics, true_pressed, true_repeated, true_dynamics)
                loss.backward()
                opt.step()

                with torch.no_grad():
                    total_loss += loss.detach()
                    total_batches += 1

                start += (bs - overlap)
                end = min(end + (bs - overlap),
                          pressed.shape[1] - 1)  # -1 so that the final prediction will have the final index
            if piece_num % 20 == 0:
                torch.save(model, path.join('model', filename))
                with torch.no_grad():
                    avg_loss = total_loss / total_batches
                    print(f'Train loss {avg_loss.item()}')

        # Evaluate validation set
        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_batches = 0
            for pressed, dynamics, repeated, beats, style in tqdm(valid):
                pressed = pressed.squeeze(0).to(torch.float32)
                dynamics = dynamics.squeeze(0).to(torch.float32)
                beats = beats.squeeze(0).to(torch.float32)
                style = style.squeeze(0).to(torch.float32)
                repeated = repeated.squeeze(0).to(torch.float32)
                piece_batches = np.ceil(
                    (pressed.shape[1] - 1 - bs) / (bs - overlap)) + 1  # Conv output formula with overlap = bs-stride
                start = 0
                end = min(bs, pressed.shape[1] - 1)
                for batch_num in range(int(piece_batches)):
                    batch_pressed = pressed[:, start:end].to(device)
                    batch_dynamics = dynamics[:, start:end].to(device)
                    batch_repeated = repeated[:, start:end].to(device)
                    batch_beats = beats[start:end].to(device)

                    true_pressed = pressed[:, start + 1 + overlap:end + 1].to(device)
                    true_repeated = repeated[:, start + 1 + overlap:end + 1].to(device)
                    true_dynamics = dynamics[:, start + 1 + overlap:end + 1].to(device)

                    pred_played, pred_dynamics = model(batch_pressed, batch_beats,
                                                       style.to(device))
                    pred_played = pred_played[:, overlap:]  # Only consider predictions that were not part of prev batch
                    pred_dynamics = pred_dynamics[:, overlap:]

                    total_loss += deepj_loss(pred_played, pred_dynamics, true_pressed, true_repeated, true_dynamics)
                    total_batches += 1

                    start += (bs - overlap)
                    end = min(end + (bs - overlap),
                              pressed.shape[1] - 1)  # -1 so that the final prediction will have the final index

            avg_loss = total_loss / total_batches
            print(f'Valid loss {avg_loss.item()}')


def generate(model: MusicStyleRelativeTransformer, device: torch.device, style: torch.Tensor,
             bpm=120, amount=128, seed=0, start: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = None,
             bs=128, temp_function: Callable[[int], float] = None, filename: str = None,
             meta_track: MidiTrack = None) -> Tuple[str, int]:
    """
    Generate music from a ``MusicStyleRelativeTransformer``, one timestep at a time.
    :param model: Model to generate music with.
    :param device: Device to run the model on.
    :param style: Style to generate with as a one-hot encoded tensor of the composer value.
    :param bpm: Beats per minute of generated output.
    :param amount: Number of timesteps to generate.
    :param seed: Seed for random generation. If set to 0, will use a random seed.
    :param start: Tuple containing the pressed, repeated, dynamics and beats values to use before generating new notes.
    :param bs: Batch size, denoting the number of previous timesteps (including the current) to consider in generation.
    :param temp_function: Function taking in the number of total notes in the last timestep to adjust temperature based on this.
    :param filename: Filename to save generated output to, in directory ``output``.
    :param meta_track: Optional meta_track to add in case of generating from existing input.
    :return:
    """
    model.eval()
    model = model.to(device)
    quantization = model.bar_quantization
    with torch.no_grad():
        if start is None:
            pressed = torch.zeros((88, 1), dtype=torch.float32).to(device)
            dynamics = torch.zeros((88, 1), dtype=torch.float32).to(device)
            repeated = torch.zeros((88, 1), dtype=torch.float32).to(device)
            beats = torch.zeros((1, quantization), dtype=torch.float32).to(device)
        else:
            pressed = torch.tensor(start[0][-bs:], dtype=torch.float32).to(device)
            dynamics = torch.tensor(start[1][-bs:], dtype=torch.float32).to(device)
            repeated = torch.tensor(start[2][-bs:], dtype=torch.float32).to(device)
            beats = torch.tensor(start[3][-bs:], dtype=torch.float32).to(device)
        quantization = 16
        beats_id = torch.eye(quantization).to(device)
        temp = 1
        silent_steps = 0
        gen = torch.Generator(device)
        if seed != 0:
            gen.manual_seed(seed)
        else:
            seed = gen.seed()

        for _ in tqdm(range(amount)):
            pred_played, pred_dynamics = model(pressed[:, -bs:], beats[-bs:],
                                               style.to(device), temp=temp)
            chosen_note = torch.bernoulli(pred_played[:, -1:, 0], generator=gen)
            chosen_repeat = torch.minimum(chosen_note, torch.bernoulli(
                pred_played[:, -1:, 1], generator=gen))  # Only add replayed note if note was chosen
            pressed = torch.cat([pressed, chosen_note], dim=1)
            repeated = torch.cat([repeated, chosen_repeat], dim=1)
            dynamics = torch.cat([dynamics, chosen_note * pred_dynamics[:, -1:]], dim=1)

            # Add new beat
            prev_beat = beats[-1].argmax()
            new_beat = (prev_beat + 1) % quantization
            beats = torch.cat([beats, beats_id[new_beat].unsqueeze(0)], dim=0)
            total_notes = chosen_note.sum()
            if total_notes == 0:
                silent_steps += 1
            else:
                silent_steps = 0
            if temp_function is not None:
                temp = temp_function(total_notes)
            else:
                temp = 0.1 * silent_steps + 1

        if meta_track is not None:
            trimmed_meta_track = MidiTrack()
            for msg in meta_track:
                if msg.type == 'time_signature':
                    trimmed_meta_track.append(msg)
                    break  # Only add first time signature message so BPM can be adjusted based on that
        else:
            trimmed_meta_track = None

        generated = decode_midi(dynamics, repeated, meta_track=trimmed_meta_track, quantization=16, bpm=bpm)

        if filename is None:
            filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.mid")
        output_path = path.join("output", filename)
        generated.save(output_path)
        return output_path, seed


def plot_style_embeddings(model: MusicStyleRelativeTransformer, device: torch.device, composers: list):
    """
    After applying 2D PCA, plots the embeddings for each composer.
    :param model: Model containing the style embeddings.
    :param device: Device to apply inference on.
    :param composers: List of composers determining labels for each point.
    """
    pca = PCA(2)
    style_id = torch.eye(len(composers))
    style_embeddings = model.style_layer(style_id[torch.arange(len(composers))].to(device))
    style_embeddings_pca = pca.fit_transform(style_embeddings.detach().cpu().numpy())
    cmap = plt.colormaps['tab20']
    for i in range(len(composers)):
        plt.scatter(style_embeddings_pca[i, 0], style_embeddings_pca[i, 1], c=cmap(i / len(composers)),
                    label=composers[i])
    plt.legend()
    plt.show()
