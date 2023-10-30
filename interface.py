import torch

import gradio as gr
from data import encode_midi, midi_to_wav, MidiFile
from model import generate, MusicStyleRelativeTransformer


def load_model(model_path: str) -> MusicStyleRelativeTransformer:
    return torch.load(model_path)


def fn_bar_quantization_text(model_state: gr.State):
    if model_state is None:
        return f"Beat quantization: N/A"
    else:
        return f"Beat quantization: 1 bar = {model_state.bar_quantization} timesteps"


def fn_amount_slider_text(model_state: gr.State, slider: gr.Slider):
    if model_state is None or slider is None:
        return None
    else:
        return f"{slider / model_state.bar_quantization * 4} beats"


def fn_generate_button(model: gr.State, amount: gr.Slider, style: gr.Dropdown, bpm: gr.Slider,
                       seed: gr.Number, soundfont: gr.Dropdown, file: gr.File, cutoff: gr.Slider):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    style_in = torch.eye(model.style_layer.in_features)[style]
    if soundfont is []:
        soundfont = None
    if file is not None:
        filename = file.name
        midi = MidiFile(filename)
        pressed, dynamics, repeated, beats, meta_track = encode_midi(midi)
        pressed = pressed[:, :cutoff]
        dynamics = dynamics[:, :cutoff]
        repeated = repeated[:, :cutoff]
        beats = beats[:cutoff]
        start = (pressed, dynamics, repeated, beats)
        # out_midi = generate(model, device, style, amount=amount, temp_function=lambda notes: 1.2*2 ** (-0.25 * notes),
        #                     start=start)
        out_midi, seed = generate(model, device, style_in, seed=seed, amount=amount,
                            start=start, bpm=bpm, meta_track=meta_track)
    else:
        # out_midi = generate(model, device, style, amount=amount, temp_function=lambda notes: 1.2*2 ** (-0.25 * notes))
        out_midi, seed = generate(model, device, style_in, seed=seed, amount=amount, bpm=bpm)

    return midi_to_wav(out_midi, soundfont), seed
