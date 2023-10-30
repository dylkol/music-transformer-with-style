import argparse
import torch

from model import train_model, MusicStyleRelativeTransformer
from data import load_data
from os import walk

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Filename of saved model (stored in ./model).", type=str)

    # Data args
    parser.add_argument("-q", "--quantization", help="Number of timesteps in bar. (default: 16)", type=int, default=16)
    parser.add_argument("-t", "--transposition", help="""
    Transpositions to augment training data, from negative to positive. 
    So, if value is 2, transpositions from -2 to +2 are added. (default: 3)
    """, type=int, default=3)
    parser.add_argument("-s", "--seed", help="Seed for shuffling of data (default: random)", type=int, default=None)
    parser.add_argument("-d", "--data_dir",
                        help="Directory where MIDI data is stored, in subdirectories for each composer. (default: ./data/)",
                        type=str, default="data")

    # Model args
    parser.add_argument("-nc", "--n_note_channels",
                        help="Number of channels for first 'note octave' conv layer. (default: 32)",
                        type=int, default=32)
    parser.add_argument("-ds", "--d_style_embeddings", help="Dimensions of style embeddings. (default: 16)",
                        type=int, default=16)
    parser.add_argument("-ntf", "--n_tf_layers", help="Number of transformer attention layers. (default: 4)",
                        type=int, default=4)
    parser.add_argument("-dfc", "--d_fc_layers",
                        help="Dimensions of FC hidden layer at the end of each attention layer. (default: 64)",
                        type=int, default=64)
    parser.add_argument("-nh", "--n_heads", help="Number of heads in transformer layer. (default: 4)",
                        type=int, default=4)
    parser.add_argument("-pd", "--p_dropout", help="Proportion of inputs to dropout in dropout layers. (default: 0.2)",
                        type=float, default=0.2)

    # Training args
    parser.add_argument("-bs", "--batch_size", help="Size in timesteps of note input. (default: 128)",
                        type=int, default=128)
    parser.add_argument("-o", "--overlap", help="""
    Number of timesteps overlap between batches.
    E.g. for bs=5, o=2 would give [0,1,2,3,4], [3,4,5,6,7] as indices of the first two batches. (default: 32)
    """, type=int, default=32)
    parser.add_argument("-e", "--epochs", help="Number of epochs to train (default: 1)", type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate for Adam optimizer (default: 1e-3)",
                        type=float, default=1e-3)
    parser.add_argument("-g", "--gpu", help="Whether to use GPU for training",
                        action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    composers = next(walk(args.data_dir))[1]
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print("GPU not available, using CPU instead.")
    else:
        device = torch.device('cpu')

    train, valid = load_data(composers, seed=args.seed, quantization=args.quantization, transposition=args.transposition)

    model = MusicStyleRelativeTransformer(d_note=args.n_note_channels,
                                          d_style=args.d_style_embeddings,
                                          tf_layers=args.n_tf_layers,
                                          n_heads=args.n_heads,
                                          d_fc=args.d_fc_layers,
                                          num_composers=len(composers),
                                          p_dropout=args.p_dropout,
                                          bar_quantization=args.quantization,
                                          max_len=args.batch_size)

    train_model(model, train, valid, device, filename=args.filename,
                lr=args.learning_rate, epochs=args.epochs, bs=args.batch_size,
                overlap=args.overlap)