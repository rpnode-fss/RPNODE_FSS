"""Training Script"""
import os
import shutil
import numpy as np
import pdb
import random

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from config import ex
from util.utils import set_seed, CLASS_LABELS, date
from dataloaders_medical.prostate import *
from models.fewshot import FewShotSeg
from models.ode import FewShotSegOde
from tqdm import tqdm
import torch.nn.functional as F

def overlay_color(img, mask, label, scale=50):
    """
    :param img: [1, 256, 256]
    :param mask: [1, 256, 256]
    :param label: [1, 256, 256]
    :return:
    """
    # pdb.set_trace()
    scale = np.mean(img.cpu().numpy())
    mask = mask[0]
    label = label[0]
    zeros = torch.zeros_like(mask)
    zeros = [zeros for _ in range(3)]
    zeros[0] = mask
    mask = torch.stack(zeros,dim=0)
    zeros[1] = label
    label = torch.stack(zeros,dim=0)
    img_3ch = torch.cat([img,img,img],dim=0)
    masked = img_3ch+mask.float()*scale+label.float()*scale
    return [masked]

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')


    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)


    _log.info('###### Create model ######')
    if _config["use_ode"]:
        model_orig = FewShotSegOde(pretrained_path=_config['path']['init_path'], pretrained_ode=_config["pretrain_ode"], ode_layers=_config["ode_layers"], ode_time=_config["ode_time"], noise_type=_config["feat_noise_type"], sigma=_config["gaussian_std"])
    else:
        model_orig = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model_orig.cuda(), device_ids=[_config['gpu_id'],])
    model.train()


    _config["data_src"] = _config["data_srcs"][_config["dataset"]]

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'BCV' or data_name == 'CTORG':
        make_data = meta_data
    else:
        print(f"data name : {data_name}")
        raise ValueError('Wrong config for dataset!')

    tr_dataset, val_dataset, ts_dataset = make_data(_config)
    print(len(tr_dataset))
    trainloader = DataLoader(
        dataset=tr_dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['n_work'],
        pin_memory=False, #True load data while training gpu
        drop_last=True
    )
    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])

    if _config['record']:  ## tensorboard visualization
        _log.info('###### define tensorboard writer #####')
        _log.info(f'##### board/train_{_config["board"]}_{date()}')
        writer = SummaryWriter(f'board/train_{_config["board"]}_{date()}')

    log_loss = {'loss': 0}
    _log.info('###### Training ######')
    total_iter = len(trainloader)

    augmentor = transforms.Compose([
        transforms.ColorJitter(),
        transforms.GaussianBlur(3, 0.1),
        # transforms.RandomErasing(0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])
    big_augmentor = transforms.Compose([
        transforms.ColorJitter(),
        transforms.GaussianBlur(3, 0.1),
        transforms.RandomEqualize(),
        transforms.RandomPosterize(2),
        transforms.RandomAdjustSharpness(1.5),
    ])
    big_augmentor = transforms.RandomApply(big_augmentor)
    # transforms.RandomAutocontrast(0.2),

    
    for i_iter, sample_batched in enumerate(tqdm(trainloader)):
        # Prepare input
        s_x_orig = sample_batched['s_x'].cuda()  # [B, Support, slice_num=1, 1, 256, 256]
        s_x = s_x_orig.squeeze(2) # [B, Support, 1, 256, 256]
        s_y_fg_orig = sample_batched['s_y'].cuda()  # [B, Support, slice_num, 1, 256, 256]
        s_y_fg = s_y_fg_orig.squeeze(2) # [B, Support, 1, 256, 256]
        s_y_fg = s_y_fg.squeeze(2) # [B, Support, 256, 256]
        s_y_bg = torch.ones_like(s_y_fg) - s_y_fg
        q_x_orig = sample_batched['q_x'].cuda()  # [B, slice_num, 1, 256, 256]
        q_x = q_x_orig.squeeze(1) # [B, 1, 256, 256]
        q_y_orig = sample_batched['q_y'].cuda()  # [B, slice_num, 1, 256, 256]
        q_y = q_y_orig.squeeze(1) # [B, 1, 256, 256]
        q_y = q_y>0
        q_y = q_y.squeeze(1).long() # [B, 256, 256]

        all_samples_s, all_samples_q = [], []
        all_labels_q = [q_y]
        
        assert _config["no_samples"] >= 0
        for _ in range(_config["no_samples"]):
            if _config["input_noise_type"] == "multiplicative":
                noise_s = torch.normal(0, _config["gaussian_std"], size=s_x.shape).cuda()
                noise_q = torch.normal(0, _config["gaussian_std"], size=q_x.shape).cuda()
                # noise_q  =  np.random.normal(0.0, scale=_config["gaussian_std"], size=q_x.shape)
                noise_s = s_x * noise_s
                noise_q = q_x * noise_q
                all_samples_s.append(s_x + noise_s)
                all_samples_q.append(q_x + noise_q)
            elif _config["input_noise_type"] == "additive":
                noise_s = torch.normal(0, _config["gaussian_std"], size=s_x.shape).cuda()
                noise_q = torch.normal(0, _config["gaussian_std"], size=q_x.shape).cuda()
                all_samples_s.append(s_x + noise_s)
                all_samples_q.append(q_x + noise_q)
            elif _config["input_noise_type"] == "augment":
                # print(s_x.shape, q_x.shape)
                init_shape = s_x.shape
                s_x_local = s_x.view(-1, init_shape[-3], init_shape[-2], init_shape[-1])
                aug_s = augmentor(s_x_local)
                aug_s = aug_s.view(init_shape)

                init_shape = q_x.shape
                q_x_local = q_x.view(-1, init_shape[-3], init_shape[-2], init_shape[-1])
                aug_q = augmentor(q_x_local)
                aug_q = aug_q.view(init_shape)
                all_samples_s.append(aug_s)
                all_samples_q.append(aug_q)
            #  elif _config["input_noise_type"] == "bigaugment":
            #     # print(s_x.shape, q_x.shape)
            #     init_shape = s_x.shape
            #     s_x_local = s_x.view(-1, init_shape[-3], init_shape[-2], init_shape[-1])
            #     aug_s = augmentor(s_x_local)
            #     aug_s = aug_s.view(init_shape)

            #     init_shape = q_x.shape
            #     q_x_local = q_x.view(-1, init_shape[-3], init_shape[-2], init_shape[-1])
            #     aug_q = augmentor(q_x_local)
            #     aug_q = aug_q.view(init_shape)
            #     all_samples_s.append(aug_s)
            #     all_samples_q.append(aug_q)
            elif _config["input_noise_type"] == "augment_and_multiplicative":
                # print(s_x.shape, q_x.shape)
                init_shape = s_x.shape
                s_x_local = s_x.view(-1, init_shape[-3], init_shape[-2], init_shape[-1])
                aug_s = augmentor(s_x_local)
                aug_s = aug_s.view(init_shape)

                init_shape = q_x.shape
                q_x_local = q_x.view(-1, init_shape[-3], init_shape[-2], init_shape[-1])
                aug_q = augmentor(q_x_local)
                aug_q = aug_q.view(init_shape)
                all_samples_s.append(aug_s)
                all_samples_q.append(aug_q)


                noise_s = torch.normal(0, _config["gaussian_std"], size=s_x.shape).cuda()
                noise_q = torch.normal(0, _config["gaussian_std"], size=q_x.shape).cuda()
                # noise_q  =  np.random.normal(0.0, scale=_config["gaussian_std"], size=q_x.shape)
                noise_s = s_x * noise_s
                noise_q = q_x * noise_q
                all_samples_s.append(s_x + noise_s)
                all_samples_q.append(q_x + noise_q)
            elif _config["input_noise_type"] == "augment_or_multiplicative":
                p = random.uniform(0, 1)
                if p > 0.5:
                    # print(s_x.shape, q_x.shape)
                    init_shape = s_x.shape
                    s_x_local = s_x.view(-1, init_shape[-3], init_shape[-2], init_shape[-1])
                    aug_s = augmentor(s_x_local)
                    aug_s = aug_s.view(init_shape)

                    init_shape = q_x.shape
                    q_x_local = q_x.view(-1, init_shape[-3], init_shape[-2], init_shape[-1])
                    aug_q = augmentor(q_x_local)
                    aug_q = aug_q.view(init_shape)
                    all_samples_s.append(aug_s)
                    all_samples_q.append(aug_q)
                else:

                    noise_s = torch.normal(0, _config["gaussian_std"], size=s_x.shape).cuda()
                    noise_q = torch.normal(0, _config["gaussian_std"], size=q_x.shape).cuda()
                    # noise_q  =  np.random.normal(0.0, scale=_config["gaussian_std"], size=q_x.shape)
                    noise_s = s_x * noise_s
                    noise_q = q_x * noise_q
                    all_samples_s.append(s_x + noise_s)
                    all_samples_q.append(q_x + noise_q)
                
            else:
                all_samples_s.append(s_x)
                all_samples_q.append(q_x)

            all_labels_q.append(q_y)
            if _config["input_noise_type"] == "augment_and_multiplicative":
                all_labels_q.append(q_y)
        # for i in range(len(all_samples_s)):
        #     all_samples_s[i] = all_samples_s[i][:, 0, ...].unsqueeze(1)
        if _config["keep_clean"]:
            all_samples_s.append(s_x)
            all_samples_q.append(q_x)
        # if _config["n_shot"]  == 3:
        #     all_samples_s = [torch.cat([all_samples_s[0], all_samples_s[1]], 1)]
        
        
        # if not _config["use_pert_prot"]:
        #     all_samples_s = [s_x]
        
        
        
        # s_xs = [[s[:,shot, ...] for shot in range(_config["n_shot"]) for s in all_samples_s]]
        s_xs = [[s_x[:,shot, ...] for shot in range(_config["n_shot"])]]
        s_y_fgs = [[s_y_fg[:,shot, ...] for shot in range(_config["n_shot"])]]
        s_y_bgs = [[s_y_bg[:,shot, ...] for shot in range(_config["n_shot"])]]
        q_y = torch.cat(all_labels_q, 0)
        q_xs = all_samples_q
        all_samples_fg  = []
        all_samples_fg += s_y_fgs
        for i in  range(_config["no_samples"]):
            # if _config["n_shot"]  == 3:
            #     all_samples_fg[0] += s_y_fgs[0]
            # else:
            all_samples_fg += s_y_fgs
        if _config["input_noise_type"] == "augment_and_multiplicative":
            for i in  range(_config["no_samples"]):
                all_samples_fg += s_y_fgs
        
        if not _config["use_cluster"]:
            all_samples_s = all_samples_s
            all_samples_fg = []

        # Forward and Backward
        optimizer.zero_grad()
        query_pred, _, all_fg_prototypes, query_feats = model(s_xs, s_y_fgs, s_y_bgs, q_xs, return_feats=True) #[B, 2, w, h]
        query_loss = criterion(query_pred, q_y)
        if len(all_samples_fg) != 0:
            # for a in all_samples_s:
            #     print(a.shape)
            pert_sup_feats, pert_supp_fg_fts = model_orig.get_sup_fore(all_samples_s, all_samples_fg)

        cluster_loss = 0
        
        # print(pert_sup_feats.shape, all_fg_prototypes.shape)
        # print(len(pert_sup_feats), len(all_fg_prototypes))
        # print(all_fg_prototypes[0].shape, pert_sup_feats[0].shape)
        # print(len(pert_sup_feats), pert_sup_feats[0].shape, len(all_fg_prototypes))
        # sim_s = [F.cosine_similarity(pert_sup_feats[j][i,  ...], all_fg_prototypes[i]).mean() for j in range(len(pert_sup_feats)) for i in range(len(all_fg_prototypes))]
        if _config["cluster_prototype"] and _config["use_cluster"]:
            sim_s = [F.cosine_similarity(pert_supp_fg_fts[i], pert_supp_fg_fts[-1]).mean() for i in range(len(pert_supp_fg_fts) -  1)]
            sim_s = sum(sim_s)/len(sim_s)#/min(len(pert_sup_feats), len(all_fg_prototypes))
            cluster_loss += 1 - sim_s

        if _config["cluster_supp"] and _config["use_cluster"]:
            sim_s = [F.cosine_similarity(pert_sup_feats[i, ...], pert_sup_feats[-1, ...]).mean() for i in range(pert_sup_feats.shape[0] -  1)]
            sim_s = sum(sim_s)/len(sim_s)#/min(len(pert_sup_feats), len(all_fg_prototypes))
            cluster_loss += 1 - sim_s

        if _config["use_cluster"]:
            sim_q = [F.cosine_similarity(query_feats[i, ...], query_feats[-1, ...]).mean() for i in range(query_feats.shape[0] -  1)]
            sim_q = sum(sim_q)/len(sim_q)#/min(len(pert_sup_feats), len(all_fg_prototypes))
            cluster_loss += 1 - sim_q

        cl_weight = _config["cluster_wt_constant"]
        if _config["cluster_weighting"]  == "linear":
            cl_weight *= (i_iter/total_iter) 

        loss = query_loss + cl_weight * cluster_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss
        query_loss = query_loss.detach().data.cpu().numpy()
        _run.log_scalar('loss', query_loss)
        log_loss['loss'] += query_loss

        # print loss and take snapshots
        if (i_iter + 1) % _config['print_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            if _config["use_cluster"]:
                cl_loss = cl_weight*cluster_loss.detach().data.cpu().numpy() / (i_iter + 1)
            else:
                cl_loss = "<not used>"
            print(f'step {i_iter+1}/{total_iter}: loss: {loss}, cl_loss: {cl_loss}')

            if _config['record']:
                batch_i = 0
                frames = []
                query_pred = query_pred.argmax(dim=1)
                query_pred = query_pred.unsqueeze(1)
                frames += overlay_color(q_x_orig[batch_i,0], query_pred[batch_i].float(), q_y_orig[batch_i,0])
                visual = make_grid(frames, normalize=True, nrow=2)
                writer.add_image("train/visual", visual, i_iter)


            print(f"train - iter:{i_iter} \t => model saved", end='\n')
            save_fname = f'{_run.observers[0].dir}/snapshots/last.pth'
            torch.save(model.state_dict(),save_fname)
            model_dir = "./model_weights_gaussian_supp"
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), model_dir + "/{}_tar{}.pth".format(_config["model_name"], _config["target"]))
            if _config["save_every"]:
                if i_iter > 3*total_iter//4:
                    cur_model_dir = os.path.join(model_dir, _config["model_name"], "tar{}".format(_config["target"]))
                    os.makedirs(cur_model_dir, exist_ok=True)
                    model_save_name = os.path.join(cur_model_dir, "{}.pth".format(i_iter + 1))
                    torch.save(model.state_dict(), model_save_name)
            
