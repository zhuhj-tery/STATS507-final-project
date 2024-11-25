import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sklearn
import sklearn.metrics
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from networks import *
import torch.nn as nn
from loss import RegressionLoss
from metric import RegressionAccuracy
from torch.nn.utils import clip_grad_norm_
# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torchmetrics import F1Score
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


class Model(object):

    def __init__(self, yaml_conf, dataloaders):

        self.dataloaders = dataloaders
        self.yaml_conf = yaml_conf
        # define model and loss
        input_dim = self.dataloaders['train'].dataset.input_dim
        output_dim = self.dataloaders['train'].dataset.output_dim
        self.def_network(yaml_conf, input_dim, output_dim, device)
        # Learning rate
        self.lr = yaml_conf["lr"]
        self.max_tau = 1.0
        self.min_tau = 0.01
        self._tau = self.max_tau - self.min_tau
        self.tau = self.max_tau

        # define optimizers
        self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr)

        # define lr schedulers
        self.exp_lr_scheduler_G = lr_scheduler.StepLR(
            self.optimizer_G, step_size=int(yaml_conf["max_num_epochs"]/2), gamma=0.1)

        # define loss function and accuracy metric
        if yaml_conf["loss_func"] == "BCE":
            if yaml_conf["DSL_mode"] == "baseline":
                self.loss_func = nn.BCELoss()
            else:
                raise ValueError(f"Loss function f{yaml_conf['loss_func']} not supported under mode {yaml_conf['DSL_mode']}")
        else:
            raise ValueError(f"Loss function {yaml_conf['loss_func']} not supported under mode {yaml_conf['DSL_mode']}")
        
        self.accuracy_metric = RegressionAccuracy(choice='r2')

        # define some other vars to record the training states
        self.running_acc = []
        self.running_loss = []
        self.train_epoch_acc = 0.0
        self.train_epoch_loss = 0.0
        self.val_epoch_acc = 0.0
        self.val_epoch_loss = 0.0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = yaml_conf["max_num_epochs"]

        self.is_training = True
        self.batch = None
        self.batch_loss = 0.0
        self.batch_acc = 0.0
        self.batch_id = 0
        self.epoch_id = 0

        self.checkpoint_dir = yaml_conf["ckpt"]
        self.checkpoint_all_dir = yaml_conf["ckpt"] + '/all_ckpt'

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.makedirs(self.checkpoint_dir)
            os.makedirs(self.checkpoint_all_dir)

        # buffers to logfile
        self.logfile = {'val_acc': [], 'train_acc': [], 'val_loss': [], 'train_loss': [], 'epochs': []}


    def def_network(self, yaml_conf, input_dim, output_dim, device):
        self.net_G = define_G(yaml_conf, input_dim=input_dim, output_dim=output_dim, device=device)


    def _load_checkpoint(self):

        if os.path.exists(os.path.join(self.checkpoint_dir, 'last_ckpt.pt')):
            print('loading last checkpoint...')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'last_ckpt.pt'))

            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])
            self.net_G.to(device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            print('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            print()

        else:
            print('training from scratch...')


    def _save_checkpoint(self, ckpt_name, save_dir=None):
        save_dir = os.path.join(save_dir, ckpt_name)
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict()
        }, save_dir)


    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()


    def _collect_running_batch_states(self):

        self.running_acc.append(self.batch_acc.item())
        self.running_loss.append(self.batch_loss.item())

        if self.is_training:
            m_batches = len(self.dataloaders['train'])
        else:
            m_batches = len(self.dataloaders['val'])

        if np.mod(self.batch_id, 100) == 1:
            print('Is_training: %s. epoch [%d,%d], batch [%d,%d], batch_loss: %.5f, running_acc: %.5f'
                  % (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m_batches,
                     self.batch_loss.item(), np.mean(self.running_acc)))


    def _collect_epoch_states(self):

        if self.is_training:
            self.train_epoch_acc = np.mean(self.running_acc).item()
            self.train_epoch_loss = np.mean(self.running_loss).item()
            print('Training, Epoch %d / %d, epoch_loss= %.5f, epoch_acc= %.5f' %
                  (self.epoch_id, self.max_num_epochs-1, self.train_epoch_loss, self.train_epoch_acc))
        else:
            self.val_epoch_acc = np.mean(self.running_acc).item()
            self.val_epoch_loss = np.mean(self.running_loss).item()
            print('Validation, Epoch %d / %d, epoch_loss= %.5f, epoch_acc= %.5f' %
                  (self.epoch_id, self.max_num_epochs - 1, self.val_epoch_loss, self.val_epoch_acc))
        print()


    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt', save_dir=self.checkpoint_dir)
        print('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)'
              % (self.val_epoch_acc, self.best_val_acc, self.best_epoch_id))
        print()

        if self.epoch_id % 50 == 0:
            self._save_checkpoint(ckpt_name=f'ckpt_{self.epoch_id}.pt', save_dir=self.checkpoint_all_dir)

        # update the best model (based on eval acc)
        if self.val_epoch_acc > self.best_val_acc:
            self.best_val_acc = self.val_epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt', save_dir=self.checkpoint_dir)
            print('*' * 10 + 'Best model updated!')
            print()


    def _update_logfile(self):

        logfile_path = os.path.join(self.checkpoint_dir, 'logfile.json')

        # read historical logfile and update
        if os.path.exists(logfile_path):
            with open(logfile_path) as json_file:
                self.logfile = json.load(json_file)

        self.logfile['train_acc'].append(self.train_epoch_acc)
        self.logfile['val_acc'].append(self.val_epoch_acc)
        self.logfile['train_loss'].append(self.train_epoch_loss)
        self.logfile['val_loss'].append(self.val_epoch_loss)
        self.logfile['epochs'].append(self.epoch_id)

        # save new logfile to disk
        with open(logfile_path, "w") as fp:
            json.dump(self.logfile, fp)


    def _visualize_prediction(self):

        batch = next(iter(self.dataloaders['val']))
        with torch.no_grad():
            self._forward_pass(batch)
        for i in range(2*3):
            plt.subplot(2, 3, i+1)
            plt.plot(self.y_true[i].cpu().numpy(), marker='.')
            plt.plot(self.y_pred[i].cpu().numpy(), marker='.')
            plt.legend(['true', 'pred'], loc=1)
        vis_path = os.path.join(self.vis_dir, 'epoch_'+str(self.epoch_id).zfill(5)+'_pred.png')
        plt.savefig(vis_path)
        plt.close()


    def _forward_pass(self, batch, evaluate_flag=False):
        self.batch = batch
        self.x = batch['x'].to(device)
        self.y_true = self.batch['y'].to(device).reshape(-1,1)
        # if torch.sum(self.y_true) < 1:
        #     return False
        self.y_pred = self.net_G(self.x)
        self.batch_acc = self.accuracy_metric(self.y_pred, self.y_true)
        self.batch_loss = self.loss_func(self.y_pred, self.y_true)
        return True

    def _save_pr_curve(self, validation_y, pre_y):
        pr, rc, pr_thresholds = sklearn.metrics.precision_recall_curve(validation_y, pre_y)
        plt.plot(rc, pr, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        logfile_path = os.path.join(self.vis_dir, f'pr_curve_{self.epoch_id}.png')
        plt.savefig(logfile_path)
        plt.close()

    def _backward_G(self):
        self.batch_loss.backward()
        # clip_grad_norm_(self.net_G.parameters(), max_norm=1, norm_type=2)


    def _clear_cache(self):
        self.running_acc = []
        self.running_loss = []

    def _update_tau(self):
        pass


    def train_models(self):

        self._load_checkpoint()

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True
            self.net_G.train()  # Set model to training mode
            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                if "max_num_batches_train" in self.yaml_conf and self.batch_id > self.yaml_conf["max_num_batches_train"]:
                    break
                trainable_flag = self._forward_pass(batch)
                if not trainable_flag:
                    continue
                # update G
                if "no_backward" in self.yaml_conf and self.yaml_conf["no_backward"] is True:
                    pass
                else:
                    self.optimizer_G.zero_grad()
                    self._backward_G()
                    self.optimizer_G.step()
                self._collect_running_batch_states()
            self._collect_epoch_states()

            self._update_tau()

            ################## Eval ##################
            ##########################################
            print('Begin evaluation...')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval() # Set model to eval mode

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                if "max_num_batches_eval" in self.yaml_conf and self.batch_id > self.yaml_conf["max_num_batches_eval"]:
                    break
                with torch.no_grad():
                    trainable_flag = self._forward_pass(batch, evaluate_flag=True)
                    if not trainable_flag:
                        continue
                self._collect_running_batch_states()
            self._collect_epoch_states()

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_checkpoints()

            ########### Update_LR Scheduler ##########
            ##########################################
            self._update_lr_schedulers()

            ############## Update logfile ############
            ##########################################
            self._update_logfile()
