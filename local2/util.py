import json
import os, sys
from collections import defaultdict

FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
import librosa
from sklearn.manifold import TSNE
from hyperparameters import hparams
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn


def populate_phonesarray(fname, feats_dir, feats_dict):
    if feats_dict is None:
        print("Expected a feature dictionary")
        sys.exit()

    feats_array = []
    f = open(fname)
    for line in f:
        line = line.split('\n')[0].split()
        feats = [feats_dict[phone] for phone in line]
    feats = np.array(feats)
    return feats


def learning_rate_decay(init_lr, global_step):
    warmup_steps = 4000.0
    step = global_step + 1.
    lr = init_lr * warmup_steps ** 0.5 * np.minimum(
        step * warmup_steps ** -1.5, step ** -0.5)
    return lr

def get_fnames(fnames_file):
    filenames_array = []
    f = open(fnames_file)
    for line in f:
        line = line.split('\n')[0]
        filenames_array.append(line)
    return filenames_array


def get_featmetainfo(desc_file, feat_name):
    f = open(desc_file)
    for line in f:
        line = line.split('\n')[0]
        feat = line.split('|')[0]
        if feat_name == feat:
            feat_length, feat_type = line.split('|')[1], line.split('|')[2]
            return feat_length, feat_type


class FloatDataSource(Dataset):
    """
    Syntax
    dataset = FloatDataSource(fnames.txt.train, etc/falcon_feats.desc, feat_name, feats_dir)
    """

    def __init__(self, fnames_file, desc_file, feat_name, feats_dir, feats_dict=None):
        self.fnames_file = fnames_file
        self.feat_name = feat_name
        self.desc_file = desc_file
        self.filenames_array = get_fnames(self.fnames_file)
        self.feat_length, self.feat_type = get_featmetainfo(self.desc_file, feat_name)
        self.feats_dir = feats_dir
        self.feats_dict = defaultdict(lambda: len(self.feats_dict)) if feats_dict is None else feats_dict

    def __getitem__(self, idx):

        fname = self.filenames_array[idx]
        if self.feat_name == 'f0':
            fname = self.feats_dir + '/' + fname.strip() + '.feats'
            feats_array = np.loadtxt(fname)

        else:
            fname = self.feats_dir + '/' + fname.strip() + '.feats.npy'
            feats_array = np.load(fname)
        return feats_array

    def __len__(self):
        return len(self.filenames_array)

    class AudioSearchDataset(object):
        def __init__(self, Mel):
            self.Mel = Mel

        def __getitem__(self, idx):
            mel = self.Mel[idx]
            idx_random = np.random.randint(len(self.Mel))
            return mel, self.Mel[idx_random]

        def __len__(self):
            return len(self.Mel)

    def collate_fn_audiosearch(batch):
        """Create batch"""

        query_length = 200  # keeping fixed for now and assuming that audio length is atleast 100
        search_audio_lengths = [len(x[0]) for x in batch]
        max_audio_len = np.max(search_audio_lengths) + 1
        # is it good to pad and then extract query? if the difference between lengths is large, padding will be really bad,
        # also should we try edge padding instead of constant padding
        search = np.array([_pad_2d(x[0], max_audio_len) for x in batch], dtype=np.float)
        min_audio_len = np.min(search_audio_lengths)
        # print("min_audio_len", min_audio_len)
        assert min_audio_len >= query_length
        query_start_idx = np.random.randint(min_audio_len - query_length)
        search_batch = torch.FloatTensor(search)
        pos_query = search_batch[:, query_start_idx: query_start_idx + query_length]

        t = 2
        query_length = np.random.randint(50, 200)
        neg_audio_lengths = [len(x[1]) for x in batch]
        max_neg_audio_len = np.max(neg_audio_lengths) + 1
        neg_query = np.array([_pad_2d(x[1], max_neg_audio_len) for x in batch], dtype=np.float)
        neg_query_batch = torch.FloatTensor(neg_query)
        min_neg_audio_len = np.min(neg_audio_lengths)
        # print("min_neg_audio_len", min_neg_audio_len)
        assert min_neg_audio_len >= query_length
        neg_queries = []
        for i in range(t):
            neg_query_start_idx = np.random.randint(min_neg_audio_len - query_length)
            neg_query = neg_query_batch[:, neg_query_start_idx: neg_query_start_idx + query_length]
            neg_queries.append(neg_query)
        return search_batch, pos_query, neg_queries

def _pad(seq, max_len):
    # print("Shape of seq: ", seq.shape, " and the max length: ", max_len)
    assert len(seq) < max_len
    # constant padding
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=0)


def _pad_2d(seq, max_len):
    # print("Shape of seq: ", seq.shape, " and the max length: ", max_len)
    assert len(seq) < max_len
    # constant padding
    x = np.pad(seq, [(0, max_len - len(seq)), (0, 0)],
               mode="constant", constant_values=0)
    return x


def data_parallel_workaround(model, input):
    device_ids = list(range(torch.cuda.device_count()))
    output_device = device_ids[0]
    replicas = torch.nn.parallel.replicate(model, device_ids)
    inputs = torch.nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = torch.nn.parallel.parallel_apply(replicas, inputs)
    y_hat = torch.nn.parallel.gather(outputs, output_device)
    return y_hat, outputs, replicas


def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))

def plot_alignment(alignment, path, info=None):
    fig, ax = plt.subplots()
    im = ax.imshow(
        alignment,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')


def save_alignment(path, attn, global_step):
    plot_alignment(attn.T, path, info="tacotron, step={}".format(global_step))


def denormalize(S):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db


def save_spectrogram(path, linear_output):
    spectrogram = denormalize(linear_output)
    plt.figure(figsize=(16, 10))
    plt.imshow(spectrogram.T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)

def inv_preemphasis(x):
    return signal.lfilter([1], [1, -hparams.preemphasis], x)


def _stft_parameters():
    n_fft = (hparams.num_freq - 1) * 2
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
    return n_fft, hop_length, win_length


def _istft(y):
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _griffin_lim(S):
    """
    librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y


def inv_spectrogram(spectrogram):
    """Converts spectrogram to waveform using librosa"""
    S = _db_to_amp(denormalize(spectrogram) + hparams.ref_level_db)  # Convert back to linear
    return inv_preemphasis(_griffin_lim(S ** hparams.power))  # Reconstruct phase

def save_states(global_step, mel_outputs, linear_outputs, attn, y,
                input_lengths, checkpoint_dir=None):
    step = str(global_step).zfill(7)
    print("Save intermediate states at step {}".format(step))

    idx = 0

    # Alignment
    path = os.path.join(checkpoint_dir, "step{}_alignment.png".format(step))

    alignment = attn[idx].cpu().data.numpy()
    save_alignment(path, alignment, step)

    # Predicted spectrogram
    path = os.path.join(checkpoint_dir, "step{}_predicted_spectrogram.png".format(step))
    linear_output = linear_outputs[idx].cpu().data.numpy()
    save_spectrogram(path, linear_output)

    # Predicted audio signal
    signal = inv_spectrogram(linear_output.T)
    path = os.path.join(checkpoint_dir, "step{}_predicted.wav".format(step))
    save_wav(signal, path)

    # Target spectrogram
    path = os.path.join(checkpoint_dir, "step{}_target_spectrogram.png".format(step))
    linear_output = y[idx].cpu().data.numpy()
    save_spectrogram(path, linear_output)

    # Target audio signal
    signal = inv_spectrogram(linear_output.T)
    path = os.path.join(checkpoint_dir, "step{}_target.wav".format(step))
    save_wav(signal, path)

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, spk_flag=None):
    step = str(step).zfill(7)
    checkpoint_path = os.path.join(
        checkpoint_dir, "checkpoint_step{}.pth".format(step))
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

    # Speaker Embedding
    if spk_flag:
        visualize_speaker_embeddings(model, checkpoint_dir, step)


#### Visualization Stuff

def visualize_phone_embeddings(model, checkpoints_dir, step):
    print("Computing TSNE")
    phone_embedding = model.embedding
    phone_embedding = list(phone_embedding.parameters())[0].cpu().detach().numpy()
    phone_embedded = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300).fit_transform(phone_embedding)

    with open(checkpoints_dir + '/ids_phones.json') as  f:
        phones_dict = json.load(f)

    ids2phones = {v: k for (k, v) in phones_dict.items()}
    phones = list(phones_dict.keys())
    y = phone_embedding[:, 0]
    z = phone_embedding[:, 1]

    fig, ax = plt.subplots()
    ax.scatter(y, z)

    for i, phone in enumerate(phones):
        ax.annotate(phone, (y[i], z[i]))

    path = checkpoints_dir + '/step' + str(step) + '_embedding_phones.png'
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()

def visualize_latent_embeddings(model, checkpoints_dir, step):
    return
    print("Computing TSNE")
    latent_embedding = model.quantizer.embedding0.squeeze(0).detach().cpu().numpy()
    num_classes = model.num_classes

    ppl_array = [5, 10, 40, 100, 200]
    for ppl in ppl_array:

        embedding = TSNE(n_components=2, verbose=1, perplexity=ppl).fit_transform(latent_embedding)

        y = embedding[:, 0]
        z = embedding[:, 1]

        fig, ax = plt.subplots()
        ax.scatter(y, z)

        for i in range(num_classes):
            ax.annotate(i, (y[i], z[i]))

        path = checkpoints_dir + '/step' + str(step) + '_latent_embedding_perplexity_' + str(ppl) + '.png'
        plt.tight_layout()
        plt.savefig(path, format="png")
        plt.close()


def return_classes(logits, dim=-1):
   _, predicted = torch.max(logits, dim)
   return predicted.view(-1).cpu().numpy()
