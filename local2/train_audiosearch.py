"""Trainining script for Tacotron speech synthesis model.

usage: train.py [options]

options:
    --conf=<json>             Path of configuration file (json).
    --gpu-id=<N>               ID of the GPU to use [default: 0]
    --exp-dir=<dir>           Experiment directory
    --checkpoint-dir=<dir>    Directory where to save model checkpoints [default: checkpoints].
    --checkpoint-path=<name>  Restore model from checkpoint path if given.
    --hparams=<parmas>        Hyper parameters [default: ].
    --log-event-path=<dir>    Log Path [default: exp/log_tacotronOne]
    -h, --help                Show this help message and exit
"""
import os, sys
from docopt import docopt

from local2.util import FloatDataSource

args = docopt(__doc__)
print("Command line args:\n", args)
gpu_id = args['--gpu-id']
print("Using GPU ", gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
import torch.nn.functional as F


from collections import defaultdict

### This is not supposed to be hardcoded #####
FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
##############################################
from tqdm import tqdm, trange
from util import *
from model import AudioFinder

import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
from sklearn.metrics import classification_report

from hyperparameters import hparams, hparams_debug_string

vox_dir ='vox'

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
use_multigpu = None

fs = hparams.sample_rate

def train(model, train_loader, val_loader, optimizer, init_lr=0.002, checkpoint_dir=None,
          checkpoint_interval=None, nepochs=None, clip_thresh=1.0):
    if use_cuda:
        model = model.cuda()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    global global_step, global_epoch
    nepochs = 200
    t = 20
    while global_epoch < nepochs:
        model.train()
        h = open(logfile_name, 'a')
        running_loss = 0.
        y_true = []
        y_pred = []
        for step, (search, pos_query, neg_query) in tqdm(enumerate(train_loader)):
            # Decay learning rate
            current_lr = learning_rate_decay(init_lr, global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()

            # Feed data
            search, pos_query, neg_query = Variable(search),  Variable( pos_query),  [Variable(x) for x in neg_query]

            if use_cuda:
                search, pos_query, neg_query = search.cuda(), pos_query.cuda(), [x.cuda() for x in neg_query]
            output_pos = model(search, pos_query)
            out = [output_pos]
            for query in neg_query:
                # out.append(output.view((-1, 1)))
                output = model(search, query)
                out.append(output)
            output_ = torch.cat(out, dim=0)
            output_classes = return_classes(output_)
            # for ce loss
            # label = torch.cuda.LongTensor(np.zeros((pos_query.shape[0]*(t+1)), float))
            # for bce loss
            label = torch.cuda.FloatTensor(np.zeros((pos_query.shape[0]*(t+1)), float))
            label[0:pos_query.shape[0]] = 1
            # Loss
            # print(label.shape, output_.shape)
            audiosearch_loss = criterion(output_, label)
            loss = audiosearch_loss

            y_true += label.tolist()
            y_pred += output_classes.tolist()

            # if global_step > 0 and global_step % hparams.save_states_interval == 0:
            #         save_states(
            #             global_step, output, None, None, label, None, checkpoint_dir)
            #         visualize_phone_embeddings(model, checkpoint_dir, global_step)

            if global_step > 0 and global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            # Update
            loss.backward(retain_graph=False)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                 model.parameters(), clip_thresh)
            optimizer.step()

            # Logs
            if global_step % 100 == 1:
                print("loss", float(loss.item()), global_step)
            # logger.log("mel loss", float(mel_loss.item()), global_step)
            global_step += 1
            running_loss += loss.item()

        averaged_loss = running_loss / (len(train_loader))
        print("loss (per epoch)", averaged_loss, global_epoch)
        print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["Not same", "Same"]))
        h.write("Loss after epoch " + str(global_epoch) + ': ' + format(running_loss / (len(train_loader))) + '\n')
        h.close()

        model.eval()
        y_true = []
        y_pred = []
        h = open(logfile_name, 'a')
        val_running_loss = 0
        for step, (search, pos_query, neg_query) in tqdm(enumerate(val_loader)):
            model.eval()
            search, pos_query, neg_query = Variable(search),  Variable( pos_query),  [Variable(x) for x in neg_query]
            if use_cuda:
                search, pos_query, neg_query = search.cuda(), pos_query.cuda(), [x.cuda() for x in neg_query]

            # Multi GPU Configuration
            if use_multigpu:
                outputs, r_, o_ = data_parallel_workaround(model, (search, pos_query, neg_query))
                mel_outputs, linear_outputs, attn = outputs[0], outputs[1], outputs[2]

            else:
                output_pos, output_neg = model(search, pos_query, neg_query)
            out = [output_pos]
            for output in output_neg:
                # out.append(output.view((-1, 1)))
                out.append(output)
            output_ = torch.cat(out, dim=0)
            output_classes = return_classes(output_)
            # for ce loss
            # label = torch.cuda.LongTensor(np.zeros((pos_query.shape[0]*(t+1)), float))
            # for bce loss
            label = torch.cuda.FloatTensor(np.zeros((pos_query.shape[0]*(t+1)), float))
            label[0:pos_query.shape[0]] = 1

            # Loss
            audiosearch_loss = criterion(output_, label)
            loss = audiosearch_loss

            y_true += label.tolist()
            y_pred += output_classes.tolist()
            global_step += 1
            val_running_loss += loss.item()

        averaged_val_loss = val_running_loss / (len(val_loader))
        print("val loss (per epoch)", averaged_val_loss, global_epoch)
        print("Val Accuracy in epoch {}:".format(global_epoch))
        print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["Not same", "Same"]))
        h.write("Loss after epoch " + str(global_epoch) + ': ' + format(averaged_val_loss / (len(val_loader))) + '\n')
        h.close()
        global_epoch += 1


if __name__ == "__main__":

    exp_dir = args["--exp-dir"]
    checkpoint_dir = args["--exp-dir"] + '/checkpoints'
    checkpoint_path = args["--checkpoint-path"]
    log_path = args["--exp-dir"] + '/tracking'
    conf = args["--conf"]
    hparams.parse(args["--hparams"])

    # Override hyper parameters
    if conf is not None:
        with open(conf) as f:
            hparams.parse_json(f.read())

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    logfile_name = log_path + '/logfile'
    h = open(logfile_name, 'w')
    h.close()

    feats_name = 'mspec'
    Mel_train = FloatDataSource(vox_dir + '/' + 'fnames.train', vox_dir + '/' + 'etc/falcon_feats.desc',
                                feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)
    Mel_val = FloatDataSource(vox_dir + '/' + 'fnames.val', vox_dir + '/' + 'etc/falcon_feats.desc',
                              feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)
    batch_size = 1
    # Dataset and Dataloader setup
    trainset = AudioSearchDataset(Mel_train)
    train_loader = data_utils.DataLoader(
        trainset, batch_size=batch_size,
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn_audiosearch, pin_memory=hparams.pin_memory)

    valset = AudioSearchDataset(Mel_val)
    val_loader = data_utils.DataLoader(
        valset, batch_size=batch_size,
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn_audiosearch, pin_memory=hparams.pin_memory)

    # Model
    model = AudioFinder()
    model = model.cuda()
    #model = DataParallelFix(model)

    optimizer = optim.Adam(model.parameters(),
                           lr=hparams.initial_learning_rate, betas=(
                               hparams.adam_beta1, hparams.adam_beta2),
                           weight_decay=hparams.weight_decay)

    # Load checkpoint
    if checkpoint_path:
        print("Load checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        try:
            global_step = checkpoint["global_step"]
            global_epoch = checkpoint["global_epoch"]
        except:
            # TODO
            pass

    # Setup tensorboard logger
    tensorboard_logger.configure(log_path)

    print(hparams_debug_string())

    # Train!
    try:
        print(hparams.nepochs)
        train(model, train_loader, val_loader, optimizer,
              init_lr=hparams.initial_learning_rate,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=hparams.nepochs,
              clip_thresh=hparams.clip_thresh)
    except KeyboardInterrupt:
        save_checkpoint(
            model, optimizer, global_step, checkpoint_dir, global_epoch)

    print("Finished")
    sys.exit(0)




