import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.backends import cudnn
from torch import optim
from tensorboardX import SummaryWriter
import sys
import os
import warnings
import numpy as np
import imageio
import random
import faulthandler
import time
import gc
from models.networks import ChartPointFlow
from args import get_args
from lib.utils import AverageValueMeter, set_random_seed, save, resume, makedirs, visualize_point_clouds, visualize_chart
from datasets import get_trainset, get_testset, init_np_seed
from pprint import pprint
import json

faulthandler.enable()

def evaluate_gen(itr, loader, model, results_mva, cd_mva, emd_mva, log, log_mva, save_dir, args):
    from metrics.evaluation_metrics import compute_all_metrics
    if not args.distributed or args.rank == 0: 
        print('---- %dth evaluation ----' % itr)
    all_sample = []
    all_ref = []

    for data in loader:
        te_pc = data['test_points']
        te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
        B, N = te_pc.size(0), te_pc.size(1)
        _, out_pc, _ = model.sample(B, N)

        # denormalize
        m, s = data['mean'].float(), data['std'].float()
        m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
        s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        all_sample.append(out_pc)
        all_ref.append(te_pc)
    
    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)

    res = compute_all_metrics(sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True)

    if itr == 1:
        results = {k: (v.cpu().detach().item()
                    if not isinstance(v, float) else v) for k, v in res.items()}
        results['itr'] = itr
        results_mva = results
        cd_mva = results['1-NN-CD-acc']
        emd_mva = results['1-NN-EMD-acc']
    else:
        results = {}
        for k, v in res.items():
            if not isinstance(v, float):
                v = v.cpu().detach().item()
            results[k] = v
            results_mva[k] = (results_mva[k] * (itr-1) + v) / itr

        results['itr'] = itr
        results_mva['itr'] = itr

        cd_mva = (cd_mva * (itr-1) + results['1-NN-CD-acc']) / itr
        emd_mva = (emd_mva * (itr-1) + results['1-NN-EMD-acc']) / itr
    
    log.write(json.dumps(results) + '\n')
    log_mva.write(json.dumps(results_mva) + '\n')
    log.flush()
    log_mva.flush()

    pprint(results_mva)

    return results_mva, cd_mva, emd_mva



def validate(loader, model, save_dir, args):
    
    
    log = open(os.path.join(save_dir,'test_gpu%s.txt' % args.gpu), 'a')
    log_mva = open(os.path.join(save_dir, 'test_mva%s.txt' % args.gpu), 'a')
    
    result_mva = {}
    cd_mva = 0
    emd_mva = 0
    for i in range(1, 16 + 1):
        result_mva, cd_mva, emd_mva = evaluate_gen(i, loader, model, result_mva, cd_mva, emd_mva, log, log_mva, save_dir, args)
    
    return cd_mva, emd_mva

def main_worker(gpu, save_dir, ngpus_per_node, init_data, args):
    # basic setup
    cudnn.benchmark = True
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.resume_checkpoint is None and os.path.exists(os.path.join(save_dir, 'checkpoint-latest.pt')):
        args.resume_checkpoint = os.path.join(save_dir, 'checkpoint-latest.pt')
        print('Checkpoint is set to the latest one.')

    # multi-GPU setup
    model = ChartPointFlow(args)
    if not args.distributed or args.rank % ngpus_per_node == 0:
        print(model)
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print("Number of parameters : {}".format(params))
    if args.distributed: 
        if args.gpu is not None:
            def _transform_(m):
                return nn.parallel.DistributedDataParallel(
                    m, device_ids=[args.gpu], output_device=args.gpu, check_reduction=True)

            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model.multi_gpu_wrapper(_transform_)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = 0
        else:
            assert 0, "DistributedDataParallel constructor should always set the single device scope"
    else:
        def _transform_(m):
            return nn.DataParallel(m)
        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    start_epoch = 1
    valid_loss_best = float("inf")
    tot_duration = 0
    optimizer = model.make_optimizer(args)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    if args.resume_checkpoint is not None:
        model, optimizer, scheduler, start_epoch, valid_loss_best, log_dir, tot_duration = resume(
            args.resume_checkpoint, model, optimizer, scheduler)
        model.set_initialized(True)
        print('Resumed from: ' + args.resume_checkpoint)
    else:
        log_dir = save_dir + "/runs/" + str(time.strftime('%Y-%m-%d_%H:%M:%S'))
        with torch.no_grad():
            init_data = init_data.to(args.gpu, non_blocking=True)
            _ = model(init_data, optimizer,  None, None, init=True)
        del init_data
        print('Actnorm is initialized')

    if not args.distributed or (args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(logdir=log_dir)
    else:
        writer = None

    # initialize datasets and loaders
    tr_dataset = get_trainset(args)
    te_dataset = get_testset(args)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(tr_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(te_dataset)
    else:
        train_sampler = None
        test_sampler = None
        
    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=True,
        worker_init_fn=init_np_seed)

    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size, shuffle=(test_sampler is None),
        num_workers=0, pin_memory=True, sampler=test_sampler, drop_last=True,
        worker_init_fn=init_np_seed)

    # save dataset statistics
    if not args.distributed or (args.rank % ngpus_per_node == 0):
        np.save(os.path.join(save_dir, "train_set_mean.npy"), tr_dataset.all_points_mean)
        np.save(os.path.join(save_dir, "train_set_std.npy"), tr_dataset.all_points_std)
        np.save(os.path.join(save_dir, "train_set_idx.npy"), np.array(tr_dataset.shuffle_idx))
    
    # main training loop
    if args.distributed:
        print("[Rank %d] World size : %d" % (args.rank, dist.get_world_size()))

    seen_inputs = next(iter(train_loader))['train_points'].cuda(args.gpu, non_blocking=True)
    unseen_inputs = next(iter(test_loader))['test_points'].cuda(args.gpu, non_blocking=True)
    del test_loader
    

    entropy_avg_meter = AverageValueMeter()
    latent_nats_avg_meter = AverageValueMeter()
    point_nats_avg_meter = AverageValueMeter()
    mutual_meter = AverageValueMeter()
    posterior_meter = AverageValueMeter()
    loss_meter = AverageValueMeter() 
    

    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    for epoch in range(start_epoch, args.epochs+1):
        start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        if epoch < args.stop_scheduler:
            scheduler.step()
            
        if writer is not None:
            writer.add_scalar('lr/optimizer', scheduler.get_lr()[0], epoch)
        
        if epoch % args.vis_freq == 0:
            save_dir_2 = os.path.join(save_dir, "images", "epoch-{:04d}".format(epoch))
            if not args.distributed or args.rank % ngpus_per_node == 0:
                makedirs(save_dir_2)
        
        if epoch % args.valid_freq == 0:
            save_dir_3 = os.path.join(save_dir, "valid", "epoch-{:04d}".format(epoch))
            if not args.distributed or args.rank % ngpus_per_node == 0:
                makedirs(save_dir_3)

        model.train()
        # train for one epoch
        
        for bidx, data in enumerate(train_loader):
            step = bidx + len(train_loader) * (epoch - 1)
            tr_batch = data['train_points']

            inputs = tr_batch.cuda(args.gpu, non_blocking=True)
            out = model(inputs, optimizer, step, writer)

            entropy, prior_nats, recon_nats, mutual, posterior, loss = out['entropy'], out['prior_nats'], out['recon_nats'], out['mutual'], out['pos'], out['loss']
            entropy_avg_meter.update(entropy)
            point_nats_avg_meter.update(recon_nats)
            latent_nats_avg_meter.update(prior_nats)
            mutual_meter.update(mutual)
            posterior_meter.update(posterior)
            loss_meter.update(loss)
            if step % args.log_freq == 0:
                duration = time.time() - start_time
                tot_duration += duration
                ave_duration = tot_duration / (step + 1)
                start_time = time.time()
                print("\r Epoch %d [%2d/%2d] Time [%3.2fs] ETA [%3.2fm] Ent %2.5f Latent %2.5f Point %2.5f Mut %2.3f Pos %2.3f loss %2.5f"
                      % (epoch, bidx, len(train_loader), duration, (args.epochs * len(train_loader) * ave_duration - tot_duration) / 60. , entropy_avg_meter.avg,
                         latent_nats_avg_meter.avg, point_nats_avg_meter.avg, mutual_meter.avg, posterior_meter.avg, loss_meter.avg), end="")
            del inputs, out
            gc.collect()
        if not args.distributed or args.rank % ngpus_per_node == 0:
            print()

        if epoch % args.valid_freq == 0 and args.valid_freq > 0:
            with torch.no_grad():
                model.eval()
                
                _, emd_mva = validate(train_loader, model, save_dir_3, args)
                valid_loss = emd_mva
                if valid_loss < valid_loss_best:
                    valid_loss_best = valid_loss
                    if not args.distributed or (args.rank % ngpus_per_node == 0):
                        save(model, optimizer, epoch + 1, scheduler, valid_loss_best, log_dir, tot_duration,
                            os.path.join(save_dir, 'checkpoint-best.pt'))
                        print('best model saved!')

        if epoch % args.save_freq == 0 and (not args.distributed or (args.rank % ngpus_per_node == 0)):
            save(model, optimizer, epoch + 1, scheduler, valid_loss_best, log_dir, tot_duration,
                os.path.join(save_dir, 'checkpoint-%d.pt' % epoch))
            save(model, optimizer, epoch + 1, scheduler, valid_loss_best, log_dir, tot_duration,
                os.path.join(save_dir, 'checkpoint-latest.pt'))
            print('model saved!')

        # save visualizations
        if epoch % args.vis_freq == 0:
            with torch.no_grad():
                # reconstructions
                model.eval()
                samples, label = model.reconstruct(unseen_inputs)
                results = []
                for idx in range(min(8, unseen_inputs.size(0))):
                    res = visualize_point_clouds(samples[idx], unseen_inputs[idx], idx,
                                                pert_order=train_loader.dataset.display_axis_order)

                    results.append(res)
                res = np.concatenate(results, axis=1)
                imageio.imwrite(os.path.join(save_dir_2, 'CPF_recon_unseen-gpu%s.png' % (args.gpu)),
                                res.transpose(1, 2, 0))
                

                samples, label = model.reconstruct(seen_inputs)
                results = []
                for idx in range(min(8, seen_inputs.size(0))):
                    res = visualize_point_clouds(samples[idx], seen_inputs[idx], idx,
                                                pert_order=train_loader.dataset.display_axis_order)

                    results.append(res)
                res = np.concatenate(results, axis=1)
                imageio.imwrite(os.path.join(save_dir_2, 'CPF_recon_seen-gpu%s.png' % (args.gpu)),
                                res.transpose(1, 2, 0))
                
                #samples
                num_samples = min(4, unseen_inputs.size(0))
                num_points = unseen_inputs.size(1)
                _, samples, label = model.sample(num_samples, num_points)
                results = []
                for idx in range(num_samples):
                    res = visualize_point_clouds(samples[idx], unseen_inputs[idx], idx,
                                                pert_order=train_loader.dataset.display_axis_order)
                    results.append(res)
                res = np.concatenate(results, axis=1)
                imageio.imwrite(os.path.join(save_dir_2,'CPF_sample-gpu%s.png' % (args.gpu)),
                                res.transpose((1, 2, 0)))
                
                for k in range(samples.shape[0]):
                    fig_filename = os.path.join(save_dir_2, 'CPF_chart_sample-gpu%s-%d.png' % (args.gpu, k))
                    visualize_chart(samples[k], label[k], fig_filename)

                print('image files saved!')


def get_init_data(args):
    tr_dataset = get_trainset(args)
    init_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=args.batch_size, shuffle=None, 
        pin_memory=True, sampler=None, drop_last=True, worker_init_fn=init_np_seed)
        
    data = next(iter(init_loader))
    inputs = data['train_points']

    return inputs


def main():
    args = get_args()
    set_random_seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        os.makedirs(os.path.join(args.save_dir, 'images'))

    with open(os.path.join(args.save_dir, 'command.sh'), 'w') as f:
        f.write('python -X faulthandler ' + ' '.join(sys.argv))
        f.write('\n')

    ngpus_per_node = torch.cuda.device_count()
    init_data = get_init_data(args)
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args.save_dir, ngpus_per_node, init_data, args))
    else:
        main_worker(args.gpu, args.save_dir, ngpus_per_node, init_data, args)


if __name__ == '__main__':
    main()
