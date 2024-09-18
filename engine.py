# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from curses import A_TOP
import math
from re import T
from typing import Iterable, Optional
import torch
from mytimm.mixup import Mixup
from mytimm.utils import accuracy, ModelEma
#from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import numpy as np
import utils
import csv
def updatePA(model,s=0.0001):
    pre_grad = None
    for name, para in model.named_parameters():
        if "PA" in name:
            #para.grad.add_(s*torch.sign(para))
            #print(para.data)
            #print(name)
            #print(para.grad)
            if para.grad != None:
                if pre_grad == None:
                    pre_grad=s*(para.data)
                    para.grad.add_(pre_grad)
                else:
                    para.grad.add_(pre_grad+s*(para.data))
                    pre_grad=para.grad
            #para.grad.add_(s*torch.sign(para.data))
            #print(para.data)

def compute_svd_orth_regu(ALL):
    regu_loss = None 
    num_param = 0 
    for A in ALL:
        A = A.sum(0)
        para_cov = A @ A.T 
        num_param += 1
        I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
        I.requires_grad = False
        if regu_loss is None:
            regu_loss = torch.norm(para_cov-I, p="fro")
        else:
            regu_loss += torch.norm(para_cov-I, p="fro") 
    return regu_loss / num_param

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False,args=None,loss_w=False,muti=False,updata=False,GA=False,gdr=-1,oldg=None,savegradient=False,Aloss=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    criterion2 = torch.nn.SmoothL1Loss()#SoftTargetCrossEntropy()
    optimizer.zero_grad()
    epoch_gradient_values = []
    for data_iter_step, (samples, targets,index) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.float().to(device, non_blocking=True)
        targets = targets.long().to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                if loss_w:
                    output,w = model(samples)
                    loss = criterion(output, targets,w)
                elif GA:
                    output,AS,AT = model(samples)
                    
                    loss = criterion(output, targets)+criterion2(AS,AT)
                elif muti:
                    output,output1,output2 = model(samples)
                    loss = criterion(output, targets)+criterion(output1, targets)+criterion(output2, targets)
                elif Aloss:
                    output,A = model(samples)
                    loss = criterion(output, targets)+compute_svd_orth_regu(A)
                else:  
                    output = model(samples)
                    loss = criterion(output, targets)
                if updata :
                    updatePA(model)
        else: # full precision
            if loss_w:
                output,w = model(samples)
                loss = criterion(output, targets,w)
                loss = loss
            elif GA:
                output,AS,AT = model(samples)
                    
                loss = criterion(output, targets)+criterion2(AS,AT)
            elif muti:
                output,output1,output2 = model(samples)
                loss = criterion(output, targets)+criterion(output1, targets)+criterion(output2, targets)+criterion((output+output1+output2)/3,targets)
            elif Aloss:
                output,A = model(samples)
                loss = criterion(output, targets)+compute_svd_orth_regu(A)
            else:
                output = model(samples)
                loss = criterion(output, targets)
            if updata :
                updatePA(model)

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)
        
        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model,epoch)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model,epoch)
        #--------------------------------------统计梯度------------------------------------------
        if savegradient:
            gradient_values = []
            for name, param in model.named_parameters():
                if 'PA' in name:
                    if param.grad != None:
                        gradient=param.grad.detach().cpu().numpy().flatten().tolist()
                        filename = 'gradients{}.csv'.format(epoch)
                        file_exists = True
                        try:
                            with open(filename, 'r') as csvfile:
                                reader = csv.reader(csvfile)
                                if not any(reader):
                                    file_exists = False
                        except FileNotFoundError:
                            file_exists = False

                        with open(filename, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            
                            writer.writerow(gradient)

            # 保存反向梯度值到NumPy文件
            # np.save('gradient_values.npy', gradient_values)
            
        #---------------------------------------------------------------------------------------
        if gdr>=0:
            gdl = np.linspace(0,gdr,9)
            i = 0
            ro = 1
            first = False
            if oldg == None:
                oldg = []
                first = True
            #cubic
            for name, param in model.named_parameters():
                if 'PA' in name:
                    #print(name)
                    if first==False:
                        parat = param.grad    
                        param.grad = param.grad + ro - 3*ro*oldg[i]**2
                        oldg[i] = param.grad 
                    else:
                        oldg.append(param.grad)
                    
                    i+=1
        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            

    # gather the stats from all processes
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},oldg

def train_one_epoch_AE(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples, targets,index) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.float().to(device, non_blocking=True)
        targets = targets.long().to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                loss = model(samples)
        else: # full precision
            loss = model(samples)
     

        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model,epoch)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model,epoch)

        torch.cuda.synchronize()


        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)

            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]

        images = images.float().to(device, non_blocking=True)
        target = target.long().to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_savefile(arg,data_loader, model, device, use_amp=False, wrong_file=None, result_file=None,ema=False):
    if wrong_file is not None:
        f_w = open(wrong_file, 'w')
    if result_file is not None:
        f_r = open(result_file, 'w')
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    loss_value = []
    score_frag = []
    label_list = []
    pred_list = []
    step = 0
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        index = batch[2]
        label_list.append(target)
        images = images.float().to(device, non_blocking=True)
        target = target.long().to(device, non_blocking=True)



        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)
        score_frag.append(output.data.cpu().numpy())
        loss_value.append(loss.data.item())
        _, predict_label = torch.max(output.data, 1)
        pred_list.append(predict_label.data.cpu().numpy())
        step += 1
        if wrong_file is not None or result_file is not None:
            predict = list(predict_label.cpu().numpy())
            true = list(target.data.cpu().numpy())
            for i, x in enumerate(predict):
                if result_file is not None:
                    f_r.write(str(x) + ',' + str(true[i]) + '\n')
                if x != true[i] and wrong_file is not None:
                    f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    score = np.concatenate(score_frag)
    score_dict = dict(
            zip(data_loader.dataset.sample_name, score))
    import pickle
    if ema :
        with open('{}/epoch{}_{}_ema_score.pkl'.format(
            arg.work_dir, 1, "test"), 'wb') as f:
            pickle.dump(score_dict, f)
    else:
        with open('{}/epoch{}_{}_score.pkl'.format(
                arg.work_dir, 1, "test"), 'wb') as f:
            pickle.dump(score_dict, f)    
    label_list = np.concatenate(label_list)
    pred_list = np.concatenate(pred_list)
    from sklearn.metrics import confusion_matrix
    confusion = confusion_matrix(label_list, pred_list)
    list_diag = np.diag(confusion)
    list_raw_sum = np.sum(confusion, axis=1)
    each_acc = list_diag / list_raw_sum
    if ema :
        with open('{}/epoch{}_{}_each_class_acc_ema.csv'.format(arg.work_dir,  1, "test"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(each_acc)
            writer.writerows(confusion) 
    else:
        with open('{}/epoch{}_{}_each_class_acc.csv'.format(arg.work_dir,  1, "test"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(each_acc)
            writer.writerows(confusion)   
        


    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate_saveskeletonfile(arg,data_loader, model, device, use_amp=False, wrong_file=None, result_file=None,ema=False,ids=[19, 22, 41, 79, 96, 100]):
    if wrong_file is not None:
        f_w = open(wrong_file, 'w')
    if result_file is not None:
        f_r = open(result_file, 'w')
    
    criterion = torch.nn.CrossEntropyLoss()
    top7 = {}
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    loss_value = []
    score_frag = []
    label_list = []
    pred_list = []
    step = 0
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        index = batch[2]
        label_list.append(target)
        images = images.float().to(device, non_blocking=True)
        target = target.long().to(device, non_blocking=True)
        ti = target.item()
        if ti not in ids:
            continue
        if ti not in top7.keys():
            top7[ti] = []
            top7["sk"+str(ti)] = []



        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
        else:
            output = model(images)
        v = output.mean(0).mean(0).mean(0)
        v = v.cpu().numpy()
        v = -v*v
        print(np.argsort(v))
        top7[ti].append(np.sort(np.argsort(v)[0:8]).tolist())
        top7["sk"+str(ti)].append(images.cpu().numpy().tolist())
        
    import json
    with open('top8new.json', 'w') as file:
        json.dump(top7, file)
