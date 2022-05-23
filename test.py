from datasets import get_testset
from args import get_args
from pprint import pprint
from metrics.evaluation_metrics import EMD_CD
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics
from collections import defaultdict
from models.networks import ChartPointFlow
import os
import torch
import numpy as np
import torch.nn as nn
import json
import re
import time
import glob

def get_test_loader(args):
    te_dataset = get_testset(args)
    loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)
    return loader

def evaluate_recon(itr, model, args, save_dir, results_mva, log, log_mva):
    print('---- %dth evaluation ----' % itr)
    cates = args.cates
    all_results = {}
    cate_to_len = {}
    for cate in cates:
        args.cates = [cate]
        loader = get_test_loader(args)

        all_sample = []
        all_ref = []
        for data in loader:
            idx_b, tr_pc, te_pc = data['idx'], data['train_points'], data['test_points']
            te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
            tr_pc = tr_pc.cuda() if args.gpu is None else tr_pc.cuda(args.gpu)
            B, N = te_pc.size(0), te_pc.size(1)
            out_pc, _ = model.reconstruct(tr_pc, num_points=N)
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
        cate_to_len[cate] = int(sample_pcs.size(0))
        
        print("Cate=%s Total Sample size:%s Ref size: %s"
            % (cate, sample_pcs.size(), ref_pcs.size()))

        # Save it
        np.save(os.path.join(save_dir, "%s_out_smp.npy" % cate),
                sample_pcs.cpu().detach().numpy())
        np.save(os.path.join(save_dir, "%s_out_ref.npy" % cate),
                ref_pcs.cpu().detach().numpy())

        results_ = EMD_CD(sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True)

        if itr == 1:
            results = {
                k: (v.cpu().detach().item() if not isinstance(v, float) else v)
                for k, v in results_.items()}
            results_mva = results
        else:
            results = {}
            for k, v in results_.items():
                if not isinstance(v, float):
                    v = v.cpu().detach().item()
                results[k] = v
                results_mva[k] = (results_mva[k] * (itr-1) + v) / itr

        pprint(results)
        all_results[cate] = results

        if itr == 5:
            print("\n" + "#" * 20 + " average 5 trials " + "#" * 20)
            print(results_mva)

    log.write(json.dumps(results) + '\n')
    log_mva.write(json.dumps(results_mva) + '\n')
    log.flush()
    log_mva.flush()

    return results_mva

def evaluate_gen(itr, model, results_mva, log, log_mva, args, save_dir):
    print('---- %dth evaluation ----' % itr)
    loader = get_test_loader(args)
    all_sample = []
    all_ref = []
 
    for data in loader:
        idx_b, te_pc = data['idx'], data['test_points']
        te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
        B, N = te_pc.size(0), te_pc.size(1)
        _, out_pc, label = model.sample(B, N)
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

    np.save(os.path.join(save_dir, "model_out_smp.npy"), sample_pcs.cpu().detach().numpy())
    np.save(os.path.join(save_dir, "model_out_ref.npy"), ref_pcs.cpu().detach().numpy())

    # Compute metrics
    metrics = compute_all_metrics(sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True)
    sample_pcl_npy = sample_pcs.cpu().detach().numpy()
    ref_pcl_npy = ref_pcs.cpu().detach().numpy()
    jsd = JSD(sample_pcl_npy, ref_pcl_npy)
    if itr == 1:
        results = {k: (v.cpu().detach().item()
                    if not isinstance(v, float) else v) for k, v in metrics.items()}
        results['JSD'] = jsd
        results['itr'] = itr
        results_mva = results
    else:
        results = {}
        for k, v in metrics.items():
            if not isinstance(v, float):
                v = v.cpu().detach().item()
            results[k] = v
            results_mva[k] = (results_mva[k] * (itr-1) + v) / itr

        results['JSD'] = jsd
        results_mva['JSD'] = (results_mva['JSD'] * (itr-1) + jsd) / itr
        results['itr'] = itr
        results_mva['itr'] = itr

    log.write(json.dumps(results) + '\n')
    log_mva.write(json.dumps(results_mva) + '\n')
    log.flush()
    log_mva.flush()

    pprint(results_mva)

    return results_mva

def main(args):
    model = ChartPointFlow(args)
    
    log_path = 'test/' + args.cates[0] + '/{}_chart'.format(args.y_dim)

    name = re.split("[/,.]", args.load_checkpoint)[-2]
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    def _transform_(m):
        return nn.DataParallel(m)

    model = model.cuda()

    print(model)
    model.multi_gpu_wrapper(_transform_)
    print("Load Path:%s" % args.load_checkpoint)
    checkpoint = torch.load(args.load_checkpoint)
    model.load_state_dict(checkpoint['model'])

    model.set_initialized(True)
    model.eval()

    with torch.no_grad():
        results_mva = {}

        start_time = time.time()
        if args.reconst_eval:
            print('evaluate reconstruction')
            log = open(os.path.join(log_path, name + '_test_recon.txt'), 'a')
            log_mva = open(os.path.join(log_path, name + '_test_mva_recon.txt'), 'a')
            for i in range(1, 5 + 1):
                results_mva = evaluate_recon(i, model, args, log_path, results_mva, log, log_mva)

        else:
            print('evaluate generation')
            log = open(os.path.join(log_path, name + '_test_gen.txt'), 'a')
            log_mva = open(os.path.join(log_path, name + '_test_mva_gen.txt'), 'a')
            for i in range(1, 16 + 1):
                results_mva = evaluate_gen(i, model, results_mva, log, log_mva, args, log_path)
                duration = time.time() - start_time
                start_time = time.time()
                print("time : {:04f}".format(duration))


if __name__ == '__main__':
    args = get_args()
    main(args)
