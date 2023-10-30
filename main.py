from interface import *
from glob import glob
from os import path, walk, listdir

if __name__ == "__main__":
    composers = next(walk('data'))[1]
    soundfonts = listdir('soundfont')

    if len(soundfonts) > 0:
        first_soundfont = soundfonts[0]
    else:
        first_soundfont = None
    model_files = list(glob(path.join("model", "*.pt")))

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                curr_model = gr.State()
                with gr.Row():
                    model_select = gr.Dropdown(model_files, label="Model file")
                    bar_quantization_text = gr.Markdown('Beat quantization: N/A')

                with gr.Row():
                    amount_slider = gr.Slider(0, 512, value=128, label="Amount of time steps to generate")
                    amount_slider_text = gr.Markdown()

                style_dropdown = gr.Dropdown(composers, type="index", value=composers[0], label="Composer style")
                bpm_slider = gr.Slider(1, 400, value=120, label="Beats per minute (4/4 time with no input MIDI)")

                soundfont_dropdown = gr.Dropdown(soundfonts, type="value", value=first_soundfont, label="Soundfont file")

                with gr.Row():
                    seed_select = gr.Number(precision=0, label="Seed for generation, zero for random seed")
                    last_seed = gr.Number(label="Last seed")
                with gr.Accordion("Initial MIDI input (optional)", open=False):
                    midi_in_select = gr.File(file_types=[".mid"], label="Starting input MIDI file")
                    with gr.Row():
                        midi_in_slider = gr.Slider(0, 512, value=128, label="Cutoff point in timesteps for above input")
                        midi_in_slider_text = gr.Markdown()
                generate_button = gr.Button("Generate")

            with gr.Row():
                audio_out = gr.Audio(type="filepath", label="Output")

        # When model changes, change the text to reflect the beat quantization
        model_select.change(load_model, model_select, curr_model)\
            .then(fn_bar_quantization_text, curr_model, bar_quantization_text)\
            .then(fn_amount_slider_text, [curr_model, amount_slider], amount_slider_text)\
            .then(fn_amount_slider_text, [curr_model, midi_in_slider], midi_in_slider_text)

        # Change the beat text to reflect the changed sliders
        amount_slider.change(fn_amount_slider_text, [curr_model, amount_slider], amount_slider_text)
        midi_in_slider.change(fn_amount_slider_text, [curr_model, midi_in_slider], midi_in_slider_text)

        generate_button.click(fn_generate_button, inputs=[
            curr_model,
            amount_slider,
            style_dropdown,
            bpm_slider,
            seed_select,
            soundfont_dropdown,
            midi_in_select,
            midi_in_slider
        ], outputs=[audio_out, last_seed])
    demo.launch()
