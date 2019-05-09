import torch

import argparse
import shutil
import os
import re

import importlib
import numpy as np


def run_config(args, config):
    run_path = 'runs/{}'.format(config['run_id'])

    # fail early, if we should train, and dir exists #######################################
    if args.train:
        if args.force:
            if os.path.exists(run_path):
                shutil.rmtree(run_path)
        try:
            directory = 'runs/{}'.format(config['run_id'])
            os.makedirs(directory)
        except Exception as e:
            print('run directory "{}" already exists'.format(directory))
            return
        torch.save(config, 'runs/{}/config.pkl'.format(config['run_id']))

    run_module = importlib.import_module(config['modules']['run']['name'])
    run = run_module.Run(config, args.cuda)

    dataloader_module = importlib.import_module(config['modules']['dataloader']['name'])

    # only load config-specified data if necessary
    if args.train or args.test or args.find_learnrate:
        dataloaders = dataloader_module.get_loaders(config)
        run.set_dataloaders(dataloaders)

    if args.find_learnrate:
        run.find_learnrate(args.min_lr, args.max_lr)
    elif args.train:
        run.save('runs/{}/initial.pkl'.format(config['run_id']))
        for i_epoch in range(config['n_epochs']):
            abort = run.advance()
            if abort:
                run.save('runs/{}/aborted.pkl'.format(config['run_id']))
                break
            else:
                run.save('runs/{}/current.pkl'.format(config['run_id']))
    elif args.test:
        if args.checkpoint is not None:
            if os.path.exists(args.checkpoint):
                run.load(args.checkpoint)
                run.test()
            else:
                print('checkpoint_filename "{}" does not exist'.format(args.checkpoint))
                exit(-1)
        else:
            print('no checkpoint specified, exiting...')
            exit(-1)
    elif args.process:
        if args.checkpoint is not None:
            if os.path.exists(args.checkpoint):
                if not os.path.exists(args.infile):
                    print('input file "{}" does not exist'.format(args.infile))
                    exit(-1)
                if os.path.exists(args.outfile):
                    print('output file "{}" does already exist'.format(args.outfile))
                    exit(-1)

                run.load(args.checkpoint)
                run.process(dataloader_module, args.infile, args.outfile)
            else:
                print('checkpoint_filename "{}" does not exist'.format(args.checkpoint))
                exit(-1)
        else:
            print('no checkpoint specified, exiting...')
            exit(-1)
    else:
        print('nothing to do specified')
        exit(-1)


def regex_in(key, regexes):
    for regex in regexes:
        match = re.match(regex, key)
        if match is not None:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('configfile')
    parser.add_argument('--run_ids', nargs='+', default=[])

    parser.add_argument('--cuda', default=False, action='store_true')
    parser.add_argument('--dry', default=False, action='store_true')
    parser.add_argument('--force', default=False, action='store_true')

    parser.add_argument('--find-learnrate', default=False, action='store_true')
    parser.add_argument('--train', default=False, action='store_true')

    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)

    parser.add_argument('--process', default=False, action='store_true')
    parser.add_argument('--infile', type=str, default=None)
    parser.add_argument('--outfile', type=str, default=None)

    parser.add_argument('--min_lr', type=float, default=1e-8)
    parser.add_argument('--max_lr', type=float, default=10.0)
    parser.add_argument('--split', nargs=2, default=[0, 1])
    args = parser.parse_args()

    # hacky hacky hacky #########################################
    configfile_contents = open(args.configfile, 'r').read()
    _globals = dict()
    _locals = dict()
    exec(configfile_contents, _globals, _locals)

    all_selected_configs = []
    for config in _locals['get_config']():
        if len(args.run_ids) == 0 or regex_in(config['run_id'], args.run_ids):
            all_selected_configs.append(config)

    n_splits = int(args.split[1])
    i_split = int(args.split[0])

    splits = np.array_split(all_selected_configs, n_splits)
    print('all selected run_ids')

    for i, split in enumerate(splits):
        print('### split {}/{} #####################'.format(i, n_splits))
        print('\n'.join([config['run_id'] for config in split]))

    print('### running split {} ################'.format(i_split))
    for config in splits[i_split]:
        if args.dry:
            print('dry: {}'.format(config['run_id']))
        else:
            print('run: {}'.format(config['run_id']))
            run_config(args, config)


main()
