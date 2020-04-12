import numpy as np

vox_dir ='vox'
feats_name = 'mspec'
train_names = vox_dir + '/' + 'fnames.train'
train_folder = vox_dir + '/' + 'festival/falcon_' + feats_name
val_names = vox_dir + '/' + 'fnames.val'
val_folder = vox_dir + '/' + 'festival/falcon_' + feats_name

audio_files = []
with open(train_names, "r") as f:
    names = f.readlines()
for name in names:
    audio = np.load(train_folder + '/' + name.rstrip() + '.feats.npy')
    if len(audio) >= 200:
        audio_files.append(name.rstrip())

with open(train_names + '_new', "w") as f:
    for name in audio_files:
        f.write(name.rstrip() + "\n")


# Traceback (most recent call last):
#   File "local/train_audiosearch.py", line 258, in <module>
#     clip_thresh=hparams.clip_thresh)
#   File "local/train_audiosearch.py", line 105, in train
#     audiosearch_loss = criterion(output, label)
#   File "/usr/local/lib/python3.5/dist-packages/torch/nn/modules/module.py", line 489, in __call__
#     result = self.forward(*input, **kwargs)
#   File "/usr/local/lib/python3.5/dist-packages/torch/nn/modules/loss.py", line 904, in forward
#     ignore_index=self.ignore_index, reduction=self.reduction)
#   File "/usr/local/lib/python3.5/dist-packages/torch/nn/functional.py", line 1970, in cross_entropy
#     return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)
#   File "/usr/local/lib/python3.5/dist-packages/torch/nn/functional.py", line 1295, in log_softmax
#     ret = input.log_softmax(dim)
# RuntimeError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
