# music-transformer-with-style

An attempt to build a music transformer for classical piano music with conditional output based on composer style. 
More or less combines the work of [Huang et al.](https://browse.arxiv.org/pdf/1809.04281.pdf) (Music Transformer) and [Mao et al.](https://browse.arxiv.org/pdf/1801.00887.pdf) (DeepJ, a style-conditioned LSTM architecture). 
Specifically, the DeepJ architecture is taken as a baseline, with the note-axis sections removed, and the time-axis replaced by a relative attention Transformer
as described by Huang et al. This means the input is taken as a piano roll, rather than using MIDI messages directly as tokens as in the music transformer.
As can be heard in the example files (in ``demo`` folder) it does not sound amazing, and this was after playing quite a bit with hyperparameters. The differences in style also do not seem very apparent.

## Requirements
- Python, everything was tested on version 3.9
- The libraries as listed in ``requirements.txt``
- Training MIDI data, for example the piano-midi data as compiled [here](https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi)
- [FluidSynth](https://github.com/FluidSynth/fluidsynth/wiki/Download), for converting MIDI files into audio.
- A [soundfont](https://github.com/FluidSynth/fluidsynth/wiki/SoundFont) to be placed in the ``soundfont`` folder.

## How to generate music
1. Clone the repository.
2. In the root repository folder, install the requirements using ``pip install -r requirements.txt`` (optionally, do this in a [virtual environment](https://docs.python.org/3/library/venv.html))
3. Run the ``main.py``, which should open an interface in your browser.
4. From this, you can select a model and composer to generate music for, along with the tempo and number of time steps (in the default model, one beat is 16 timesteps, independent of time signature)
5. Optionally, one can use an input midi file to act as the initial data before generation, with a given cutoff point from which to start the generation. 
Generally works better than generating from nothing but it should still be very apparent where the real music ends and the generation starts.

## Training a new model
1. Do steps 1 and 2 from above.
2. Put the desired training data in the `data` folder, in a subfolder for each composer. See the [data from piano-midi.de](https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi).
3. Run ``train.py <name>`` with `name` denoting the desired model output filename, which will be stored in `./model/name.pt`. Use the ``--help`` argument for explanation of all other optional arguments. 
Under the default hyperparameters a model trains reasonably fast, but sounds pretty terrible. What I think also makes a difference is the dropout value, this is at 0.5 in DeepJ but it seems here keeping it low won't overfit much and give better results. See also the hyperparameters of the demo model below.
4. After training the model (will probably take a while) it should show up on the interface for generation when you run `main.py`.

## Hyperparameters demo model
The model was trained on the data from piano-mide.de as available [here](https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi)
| Parameter      | Value |
|----------------|-------|
| `quantization`   | 16    |
| `transposition`  | 3     |
| `seed`           | 1     |
| `n_note_channels` | 48    |
| `d_style_embeddings` | 64 |
| `d_fc_layers` | 256 |
| `n_heads` |  8 |
| `p_dropout` | 0.1 |
| `batch_size` | 128 |
| `overlap` | 16 |
| `epochs` | 2 |
| `learning_rate` | 1e-3 | 
