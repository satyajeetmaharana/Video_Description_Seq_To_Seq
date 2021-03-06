import json
import os

import numpy as np

import misc.utils as utils
import opts
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from dataloader import VideoDataset
from misc.rewards import get_self_critical_reward, init_cider_scorer
from models import DecoderRNN, EncoderRNN, EncoderDecoderModel
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

dataset_val =  None
dataloader_val = None

def train(loader, model, crit, optimizer, lr_scheduler, opt, rl_crit=None):
    model.train()
    #model = nn.DataParallel(model, device_ids=[0, 1, 2])
    for epoch in tqdm(range(opt["epochs"])):
        iteration = 0
        model.train()
        for data in loader:
            torch.cuda.synchronize()
            fc_feats = data['fc_feats'].squeeze(dim=1).cuda()
            labels = data['labels'].cuda()
            masks = data['masks'].cuda()
            optimizer.zero_grad()
            seq_probs, _ = model(fc_feats, labels, 'train')
            loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            lr_scheduler.step()
            iteration += 1
            if iteration % 15 == 0:
                print("\n iter %d (epoch %d), train_loss = %.6f" %(iteration, epoch, train_loss))
        val_loss = 0
        model.eval()
        for val_data in dataloader_val:
            torch.cuda.synchronize()
            fc_feats = val_data['fc_feats'].squeeze(dim=1).cuda()
            labels = val_data['labels'].cuda()
            masks = val_data['masks'].cuda()
            seq_probs, _ = model(fc_feats, labels, 'train')
            loss = crit(seq_probs, labels[:, 1:], masks[:, 1:])
            val_loss += loss.item()
            torch.cuda.synchronize()
            
        print("\n(epoch %d), val_loss = %.6f" %(epoch, val_loss))
        
        if epoch % 50 == 0:
            model_path = os.path.join(opt["checkpoint_path"],'model_%d.pth' % (epoch))
            model_info_path = os.path.join(opt["checkpoint_path"],'model_score.txt')
            torch.save(model.state_dict(), model_path)
            print("model saved to %s" % (model_path))
            with open(model_info_path, 'a') as f:
                f.write("model_%d, loss: %.6f , val loss: %.6f\n" % (epoch, train_loss,val_loss))


def main(opt):
    dataset = VideoDataset(opt, 'train')
    dataloader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    global dataset_val
    global dataloader_val
    dataset_val =  VideoDataset(opt, 'val')
    dataloader_val = DataLoader(dataset_val, batch_size=opt["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    opt["vocab_size"] = dataset.get_vocab_size()
    encoder = EncoderRNN(
        opt["dim_vid"],
        opt["dim_hidden"],
        bidirectional=bool(opt["bidirectional"]),
        input_dropout_p=opt["input_dropout_p"],
        rnn_cell=opt['rnn_type'],
        rnn_dropout_p=opt["rnn_dropout_p"])
    decoder = DecoderRNN(
        opt["vocab_size"],
        opt["max_len"],
        opt["dim_hidden"],
        opt["dim_word"],
        input_dropout_p=opt["input_dropout_p"],
        rnn_cell=opt['rnn_type'],
        rnn_dropout_p=opt["rnn_dropout_p"],
        bidirectional=bool(opt["bidirectional"]))
    model = EncoderDecoderModel(encoder, decoder)
    model = model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('data/save_vatex_batch_noc3d/model_500.pth'))
    crit = utils.LanguageModelCriterion()
    optimizer = optim.Adam(model.parameters(),lr=opt["learning_rate"],weight_decay=opt["weight_decay"])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=opt["learning_rate_decay_every"],gamma=opt["learning_rate_decay_rate"])
    print("Data Loaded")
    train(dataloader, model, crit, optimizer, exp_lr_scheduler, opt, rl_crit)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    if not os.path.isdir(opt["checkpoint_path"]):
        os.mkdir(opt["checkpoint_path"])
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))
    
    main(opt)
