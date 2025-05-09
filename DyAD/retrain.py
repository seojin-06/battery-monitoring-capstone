import json
import os
import pickle
import sys
import time


from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import torch.nn.utils.prune as prune

import torch.nn as nn

from tqdm import tqdm
from model import tasks
from model import dynamic_vae
from utils import to_var, collate, Normalizer, PreprocessNormalizer
from model import dataset


class Train_fivefold:
    """
    for training
    """

    def __init__(self, args, fold_num=0, model_path=None):
        """
        initialization, load project arguments and create folders
        :param model_path: 학습을 이어서 진행할 모델이 저장된 경로
        """
        self.args = args
        time_now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        current_path = os.path.join(self.args.save_model_path, time_now+'_fold%d'%fold_num)
        self.mkdir(current_path)
        self.current_path = current_path
        self.current_epoch = 1
        self.step = 1
        self.loss_dict = OrderedDict()
        self.fold_num = fold_num

        loss_picture_path = os.path.join(current_path, "loss")
        feature_path = os.path.join(current_path, "feature")
        current_model_path = os.path.join(current_path, "model")
        save_feature_path = os.path.join(current_path, "mean")
        result_path = os.path.join(current_path, "result")
        # create folders
        self.mkdir(loss_picture_path)
        self.mkdir(feature_path)
        self.mkdir(current_model_path)
        self.mkdir(result_path)
        self.mkdir(save_feature_path)

        self.args.loss_picture_path = loss_picture_path
        self.args.feature_path = feature_path
        self.args.result_path = result_path
        self.args.save_feature_path = save_feature_path
        self.args.current_path = current_path
        self.args.current_model_path = current_model_path
        
        
        # 모델 경로가 주어지면 기존 모델을 불러옴
        self.model = None
        if model_path and model_path != "":
            self.model = self.load_model_from_path(model_path)
        else:
            print("No pre-trained model found, initializing a new model.")
            self.model = self.initialize_model()

        """    
        # 모델 경량화 (프루닝 및 양자화 적용)
        self.apply_model_compression()
        """

    def apply_model_compression(self):
        self.model = self.apply_pruning(self.model, pruning_amount=0.4)
        
    
    def apply_pruning(self, model, pruning_amount=0.4):
        """
        모델에 프루닝을 적용하는 함수
        :param model: 프루닝을 적용할 모델
        :param pruning_amount: 프루닝 비율
        :return: 프루닝된 모델
        """
        # RNN 계층만 프루닝
        prune.random_unstructured(model.encoder_rnn, name="weight_ih_l0", amount=pruning_amount)
        prune.random_unstructured(model.encoder_rnn, name="weight_hh_l0", amount=pruning_amount)
        prune.random_unstructured(model.decoder_rnn, name="weight_ih_l0", amount=pruning_amount)
        prune.random_unstructured(model.decoder_rnn, name="weight_hh_l0", amount=pruning_amount)

        # RNN의 backward 방향도 pruning
        prune.random_unstructured(model.encoder_rnn, name="weight_ih_l0_reverse", amount=pruning_amount)
        prune.random_unstructured(model.encoder_rnn, name="weight_hh_l0_reverse", amount=pruning_amount)
        prune.random_unstructured(model.decoder_rnn, name="weight_ih_l0_reverse", amount=pruning_amount)
        prune.random_unstructured(model.decoder_rnn, name="weight_hh_l0_reverse", amount=pruning_amount)

        # 프루닝 후 0인 파라미터 제거
        prune.remove(model.encoder_rnn, 'weight_ih_l0')
        prune.remove(model.encoder_rnn, 'weight_hh_l0')
        prune.remove(model.decoder_rnn, 'weight_ih_l0')
        prune.remove(model.decoder_rnn, 'weight_hh_l0')

        prune.remove(model.encoder_rnn, 'weight_ih_l0_reverse')
        prune.remove(model.encoder_rnn, 'weight_hh_l0_reverse')
        prune.remove(model.decoder_rnn, 'weight_ih_l0_reverse')
        prune.remove(model.decoder_rnn, 'weight_hh_l0_reverse')
        # 프루닝된 레이어에 대해서만 동결
        for name, param in model.named_parameters():
            if "encoder_rnn" in name or "decoder_rnn" in name:
                param.requires_grad = False

        print("Pruning applied to model layers.")
        return model
        


    @staticmethod
    def mkdir(path):
        """
        create folders
        :param path: path
        """
        if os.path.exists(path):
            print('%s is exist' % path)
        else:
            os.makedirs(path)


    def initialize_model(self):
        """
        새로운 모델을 초기화하는 함수
        :return: 새로 초기화된 모델
        """
        params = dict(
            rnn_type=self.args.rnn_type,
            hidden_size=self.args.hidden_size,
            latent_size=self.args.latent_size,
            num_layers=self.args.num_layers,
            bidirectional=self.args.bidirectional,
            kernel_size=self.args.kernel_size,
            nhead=self.args.nhead,
            dim_feedforward=self.args.dim_feedforward,
            variable_length=self.args.variable_length,
            encoder_embedding_size=self.data_task.encoder_dimension,
            decoder_embedding_size=self.data_task.decoder_dimension,
            output_embedding_size=self.data_task.output_dimension)
        
        if self.args.model_type == "rnn":
            model = to_var(dynamic_vae.DynamicVAE(**params)).float()
        else:
            model = None
        return model


    def load_model_from_path(self, model_path):
        """
        저장된 모델을 경로에서 불러오는 함수
        :param model_path: 모델 경로
        :return: 불러온 모델
        """
        model_file = os.path.join(model_path, "model.torch")
        if os.path.exists(model_file):
            print(f"Resuming from saved model: {model_file}")
            model = torch.load(model_file)
            return model
        else:
            print("No saved model found at the specified path.")
            return None
        

    def main(self):
        """
        training
        load training data, preprocessing, create & train & save model, save parameters
        train: normalized data
        model: model
        loss: nll kl label
        rec_error: reconstruct error
        """
        print("Loading data to memory. This may take a few minutes...")
        data_pre = dataset.Dataset(self.args.train_path, train=True, fold_num=self.fold_num)
        self.normalizer = Normalizer(dfs=[data_pre[i][0] for i in range(200)],
                                     variable_length=self.args.variable_length)
        train = PreprocessNormalizer(data_pre, normalizer_fn=self.normalizer.norm_func)
        print("Data loaded successfully.")

        self.args.columns = torch.load(os.path.join(os.path.dirname(self.args.train_path), "column.pkl"))
        self.data_task = tasks.Task(task_name=self.args.task, columns=self.args.columns)

        model = self.model

        print("model", model)
        
        # 전체 파라미터 개수 계산
        num_params = sum(p.numel() for p in model.parameters())
        print(f"모델 총 파라미터 개수: {num_params:,}")


        """
        nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
        total_params = sum(p.numel() for p in model.parameters())

        sparsity = 100 * (1 - nonzero_params / total_params)
        print(f"Pruned sparsity: {sparsity:.2f}%")
        """

        #pytorch ver.
        def print_model_size(mdl):
            torch.save(mdl.state_dict(), "tmp.pt")
            print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
            os.remove('tmp.pt')

        print_model_size(model)
        
        
        # specify optimizer and learning scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs,
                                      eta_min=self.args.cosine_factor * self.args.learning_rate)
        # DataLoader
        data_loader = DataLoader(dataset=train, batch_size=self.args.batch_size, shuffle=True,
                                 num_workers=self.args.jobs, drop_last=False, pin_memory=torch.cuda.is_available(),
                                 collate_fn=collate if self.args.variable_length else None)
        time_start = time.time()
        try:
            p_bar = tqdm(total=len(data_loader) * self.args.epochs, desc='training', ncols=160, mininterval=1,
                         maxinterval=10, miniters=1)
            while self.current_epoch <= self.args.epochs:
                model.train()
                total_loss, total_nll, total_label, total_kl, iteration = 0, 0, 0, 0, 0
                for batch in data_loader:
                    batch_ = to_var(batch[0]).float()
                    seq_lengths = batch[1]['seq_lengths'] if self.args.variable_length else None
                    log_p, mean, log_v, z, mean_pred = model(batch_,
                                                             encoder_filter=self.data_task.encoder_filter,
                                                             decoder_filter=self.data_task.decoder_filter,
                                                             seq_lengths=seq_lengths, noise_scale=self.args.noise_scale)
                    target = self.data_task.target_filter(batch_)

                    nll_loss, kl_loss, kl_weight = self.loss_fn(log_p, target, mean, log_v)
                    self.label_data = tasks.Label(column_name="mileage", training_set=train)
                    label_loss = self.label_data.loss(batch, mean_pred, is_mse=True)
                    loss = (self.args.nll_weight * nll_loss + self.args.latent_label_weight * label_loss + kl_weight *
                            kl_loss / batch_.shape[0])

                    # update parameters
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # calculate loss
                    total_loss += loss.item()
                    total_nll += nll_loss.item()
                    total_label += label_loss.item()
                    total_kl += kl_loss.item() / batch_.shape[0]
                    loss_info = {'mean_loss': total_loss / (1 + iteration), 'nll_loss': total_nll / (1 + iteration),
                                 "label_loss": total_label / (1 + iteration), "kl_loss": total_kl / (1 + iteration)}
                    p_bar.set_postfix(loss_info)
                    p_bar.set_description('training - Epoch %d/%i' % (self.current_epoch, self.args.epochs))

                    # save loss
                    if iteration == len(data_loader) - 1:
                        self.save_loss(loss_info, log_p, target)

                    self.step += 1
                    p_bar.update(1)
                    iteration += 1

                scheduler.step()
                self.current_epoch += 1
            p_bar.close()

        except KeyboardInterrupt:
            print("Caught keyboard interrupt; quit training.")
            pass

        print("Train completed, save information")
        
        self.model_result_save(model)
        
        # save model and parameters
        model.eval()
        p_bar = tqdm(total=len(data_loader), desc='saving', ncols=100, mininterval=1, maxinterval=10, miniters=1)
        extract(data_loader, model, self.data_task, self.args.feature_path, p_bar, self.args.noise_scale,
                self.args.variable_length)
        p_bar.close()
        print("The total time consuming: ", time.time() - time_start)
        self.model_result_save(model)
        self.loss_visual()
        print("All parameters have been saved at", self.args.feature_path)

    def model_result_save(self, model):
        """
        save model
        :param model: vae or transformer
        :return:
        """
        model_params = {'train_time_start': self.current_path,
                        'train_time_end': time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                        'args': vars(self.args),
                        'loss': self.loss_dict}
        with open(os.path.join(self.args.current_model_path, 'model_params.json'), 'w') as f:
            json.dump(model_params, f, indent=4)
        model_path = os.path.join(self.args.current_model_path, "model.torch")
        torch.save(model, model_path)
        
        norm_path = os.path.join(self.args.current_model_path, "norm.pkl")
        with open(norm_path, "wb") as f:
            pickle.dump(self.normalizer, f)

    def loss_fn(self, log_p, target, mean, log_v):
        """
        loss function
        :param log_p: transformed prediction
        :param target: target
        :param mean:
        :param log_v:
        :return: nll_loss, kl_loss, kl_weight
        """
        nll = torch.nn.SmoothL1Loss(reduction='mean')
        nll_loss = nll(log_p, target)
        kl_loss = -0.5 * torch.sum(1 + log_v - mean.pow(2) - log_v.exp())
        """
        suppose
        kl_loss = -0.5 * torch.sum(1 + log_v - mean.pow(2) - torch.exp(log_v + 1e-8))
        """
        kl_weight = self.kl_anneal_function()
        return nll_loss, kl_loss, kl_weight

    def kl_anneal_function(self):
        """
        anneal update function
        """
        if self.args.anneal_function == 'logistic':
            return self.args.anneal0 * float(1 / (1 + np.exp(-self.args.k * (self.step - self.args.x0))))
        elif self.args.anneal_function == 'linear':
            return self.args.anneal0 * min(1, self.step / self.args.x0)
        else:
            return self.args.anneal0

    def loss_visual(self):
        """
        draw loss curves
        """
        if self.args.epochs == 0:
            return
        x = list(self.loss_dict.keys())
        df_loss = pd.DataFrame(dict(self.loss_dict)).T.sort_index()
        mean_loss = df_loss['mean_loss'].values.astype(float)
        nll_loss = df_loss['nll_loss'].values.astype(float)
        label_loss = df_loss['label_loss'].values.astype(float)
        kl_loss = df_loss['kl_loss'].values.astype(float)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(x, mean_loss, 'r.-', label='mean_loss')
        plt.legend()

        plt.subplot(2, 3, 4)
        plt.plot(x, nll_loss, 'bo-', label='nll_loss')
        plt.legend()

        plt.subplot(2, 3, 5)
        plt.plot(x, label_loss, 'bo-', label='label_loss')
        plt.legend()

        plt.subplot(2, 3, 6)
        plt.plot(x, kl_loss, 'bo-', label='kl_loss')
        plt.legend()
        plt.savefig(self.args.loss_picture_path + '/' + 'loss.png')
        plt.close('all')

    def save_loss(self, loss_info, log_p, target):
        """
        save loss
        """
        self.loss_dict[str(self.current_epoch)] = loss_info
        n_image = log_p.shape[-1]
        for i in range(n_image):
            plt.subplot(n_image, 1, i + 1)
            plt.plot(log_p[0, :, i].cpu().detach().numpy(), 'y',
                     label='lp-' + str(self.current_epoch))
            plt.plot(target[0, :, i].cpu().detach().numpy(), 'c',
                     label='tg-' + str(self.current_epoch))
            plt.legend()
        loss_path = os.path.join(self.args.loss_picture_path, "%i_epoch.jpg" % self.current_epoch)
        plt.savefig(loss_path)
        plt.close('all')

    def getmodelparams(self):
        return os.path.join(self.args.current_model_path, 'model_params.json')


def save_features_info(feature_path, batch, iteration, log_p, mean, target):
    """
    save features
    """
    mse = torch.nn.MSELoss(reduction='mean')
    dict_path = os.path.join(feature_path, "%i_label.file" % iteration)
    with open(dict_path, "wb") as f:
        rec_error = [float(mse(log_p[i], target[i])) for i in range(batch[0].shape[0])]
        batch[1].update({'rec_error': rec_error})
        torch.save(batch[1], f)
    mean_path = os.path.join(feature_path, "%i_npy.npy" % iteration)
    np_mean = mean.data.cpu().numpy()
    np.save(mean_path, np_mean)


def extract(data_loader, model, data_task, feature_path, p_bar, noise_scale, variable_length):
    """
    extract features
    """
    iteration = 0
    for batch in data_loader:
        batch_ = to_var(batch[0]).float()
        seq_lengths = batch[1]['seq_lengths'] if variable_length else None
        log_p, mean, log_v, z, mean_pred = model(batch_, encoder_filter=data_task.encoder_filter,
                                                 decoder_filter=data_task.decoder_filter,
                                                 seq_lengths=seq_lengths, noise_scale=noise_scale)
        target = data_task.target_filter(batch_)
        # print(log_p.shape, target.shape)  # torch.Size([64, 128, 4]) torch.Size([64, 128, 4])
        save_features_info(feature_path, batch, iteration, log_p, mean, target)
        p_bar.update(1)
        iteration += 1

