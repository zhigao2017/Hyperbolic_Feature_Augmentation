import argparse
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from dataloader.samplers import CategoriesSampler
from models.protonet_ourtrans import ProtoNet
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, compute_confidence_interval

def load(model_sv, name=None):
    if name is None:
        name = 'model'
    model = make(model_sv[name], **model_sv[name + '_args'])
    model.load_state_dict(model_sv[name + '_sd'])
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--dim', type=int, default=640)
    parser.add_argument('--curvaturedim', type=int, default=32)
    parser.add_argument('--rho', type=int, default=64)
    parser.add_argument('--r', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query_num', type=int, default=15)
    parser.add_argument('--train_way', type=int, default=5)
    parser.add_argument('--validation_way', type=int, default=5)
    parser.add_argument('--sample_num', type=int, default=5)
    parser.add_argument('--augment_lambda', type=float, default=1)
    parser.add_argument('--train_step', type=int, default=10)
    parser.add_argument('--multi_step_loss_num_epochs', type=int, default=20)
    parser.add_argument('--metalr', type=float, default=0.001)
    parser.add_argument('--innerlr', type=float, default=0.001)
    parser.add_argument('--temperature', type=float, default=100)
    parser.add_argument('--dataset', type=str, default='CUB', choices=['MiniImageNet', 'CUB'])
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--lr_decay', type=bool, default=True)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--p_decay', default=0.0005,   type=float, help='Weight decay for optimizer.')
    parser.add_argument('--step_size', type=int, default=40)
    parser.add_argument('--curvature', type=float, default=-0.1)
    parser.add_argument('--curvaturel', type=float, default=0.1)
    parser.add_argument('--curvaturescale', type=float, default=0.1)
    parser.add_argument('--curvaturestart', type=float, default=0.1)
    parser.add_argument('--tradeoff1', type=float, default=1)
    parser.add_argument('--tradeoff2', type=float, default=1)
    args = parser.parse_args()
    pprint(vars(args))

    if torch.cuda.is_available():
        print('CUDA IS AVAILABLE')
#     set_gpu(args.gpu)


    if args.save_path is None:
        save_path1 = '-'.join([args.dataset, 'ProtoNet'])
        save_path2 = '_'.join([str(args.shot), str(args.query_num), str(args.train_way), str(args.validation_way),
                               str(args.metalr), str(args.innerlr),
                               str(args.dim)])
        args.save_path = save_path1 + '_' + save_path2
        ensure_path(args.save_path)
    else:
        ensure_path(args.save_path)

    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from dataloader.cub import CUB as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, 100, args.train_way, args.shot + args.query_num)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)



    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.label, 1000, args.validation_way, args.shot + 15)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=8, pin_memory=True)

    model = ProtoNet(args)


    print(model)
    
    small_p_decay=args.p_decay*1


    to_optim          = [{'params':model.controller.parameters(),'lr':args.metalr,'weight_decay':small_p_decay},
                         {'params':model.de.parameters(),'lr':args.metalr,'weight_decay':small_p_decay},
                         {'params':model.ma.parameters(),'lr':args.metalr,'weight_decay':small_p_decay},
                         {'params':model.ca.parameters(),'lr':args.metalr,'weight_decay':small_p_decay},
                         {'params':model.cura.parameters(),'lr':args.metalr,'weight_decay':small_p_decay},
                         {'params':model.cd.parameters(),'lr':args.metalr,'weight_decay':small_p_decay},
                         {'params':model.proto_attn.parameters(),'lr':args.metalr,'weight_decay':small_p_decay},                         
                         {'params':model.mean_mean,'lr':args.metalr,'weight_decay':small_p_decay},
                         {'params':model.mean_var,'lr':args.metalr,'weight_decay':small_p_decay},
                         {'params':model.var_mean,'lr':args.metalr,'weight_decay':small_p_decay},
                         {'params':model.var_var,'lr':args.metalr,'weight_decay':small_p_decay},
                         #{'params':model.tangentpoint,'lr':args.metalr,'weight_decay':small_p_decay},
                         {'params':model.lambda_0,'lr':args.metalr},
                         {'params':model.lambda_1,'lr':args.metalr},
                         ]

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.Adam(to_optim)
    optimizer = torch.optim.RMSprop(to_optim)



    if args.lr_decay:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)



    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    



    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))


    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['test_loss'] = []
    trlog['train_acc'] = []
    trlog['test_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    timer = Timer()
    global_count = 0
    writer = SummaryWriter(comment=args.save_path)

    for epoch in range(1, args.max_epoch + 1):

        if args.lr_decay:
            lr_scheduler.step()

        model.eval()
        vl = Averager()
        va = Averager()

        label = torch.arange(args.validation_way).repeat(15)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)

        test_query_acc=torch.zeros(args.train_step+1)
        print('best epoch {}, best test acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
        for i, batch in enumerate(test_loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            p = args.shot * args.validation_way
            data_shot, data_query = data[:p], data[p:]
            logits, query_acc = model(data_shot, data_query, epoch, label)
            logits=logits.detach()
            query_acc=query_acc.detach()
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            vl.add(loss.item())
            va.add(acc)
            test_query_acc=test_query_acc+query_acc

        test_query_acc=test_query_acc/1000
        #print('test accuracy array', test_query_acc)
        test_acc=torch.max(test_query_acc).item()

        vl = vl.item()
        va = va.item()
        writer.add_scalar('data/test_loss', float(vl), epoch)
        writer.add_scalar('data/test_acc', float(test_acc), epoch)
        print('epoch {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, test_acc))

        if test_acc > trlog['max_acc']:
            trlog['max_acc'] = test_acc
            trlog['max_acc_epoch'] = epoch
            save_model('max_acc')

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))


        ###Model train
        model.train()
        tl = Averager()
        ta = Averager()

        label = torch.arange(args.train_way).repeat(args.query_num)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
        epoch_totalloss=0
        epoch_accloss=0
        epoch_discloss=0
        epoch_variousloss=0
        epoch_normloss=0
        epoch_metaloss=0

        for i, batch in enumerate(train_loader, 1):
            global_count = global_count + 1
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]
            logits, query_logit_sample, loss_diversity, norm_loss, loss_meta = model(data_shot, data_query, epoch, label)
            acc_loss=F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            
            loss_regular=0
            for jj in range(args.sample_num):
                loss_regular=loss_regular+F.cross_entropy(query_logit_sample[jj,:,:], label)
            loss_regular=loss_regular/(args.sample_num+1)
            
            epoch_accloss=epoch_accloss+acc_loss.detach()
                
            loss=loss_meta+args.tradeoff1*loss_diversity+args.tradeoff2*loss_regular
            epoch_totalloss=epoch_totalloss+loss.detach()
            epoch_discloss=epoch_discloss+loss_regular.detach()
            epoch_variousloss=epoch_variousloss+loss_diversity.detach()
            epoch_metaloss=epoch_metaloss+loss_meta

            writer.add_scalar('data/loss', float(loss), global_count)
            writer.add_scalar('data/acc', float(acc), global_count)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))
            print('acc loss:',acc_loss.detach().cpu().numpy(),', various loss:',loss_diversity.detach().cpu().numpy(), ', discriminative loss:', loss_regular.detach().cpu().numpy(), 'meta accloss:', loss_meta.detach().cpu().numpy())

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()

            
            optimizer.step()
        tl = tl.item()
        ta = ta.item()
        print('epoch,',epoch,' epoch_accloss,', epoch_accloss.detach().cpu().numpy(),  ' epoch_variouscloss,', epoch_variousloss.detach().cpu().numpy(),' epoch_discriminativecloss,', epoch_discloss.detach().cpu().numpy(),' epoch_metaloss,', epoch_metaloss.detach().cpu().numpy())

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['test_loss'].append(vl)
        trlog['test_acc'].append(test_acc)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')


    writer.close()
