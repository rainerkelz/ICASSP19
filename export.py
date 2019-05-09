import torch
from utils import stacked_dict
from torch.utils.data.dataloader import DataLoader, SequentialSampler
import argparse
import os
import numpy as np
import importlib


def predict_one_sequence(args, config, run, sequence):
    run.net.eval()
    loader = DataLoader(
        sequence,
        batch_size=config['batchsize'],
        sampler=SequentialSampler(sequence),
        num_workers=0,
        pin_memory=False
    )

    predictions = []
    # batches = []
    with torch.no_grad():
        for batch in loader:
            if args.cuda:
                for key in batch.keys():
                    batch[key] = batch[key].cuda()

            _prediction = run.net.forward(batch)
            prediction = dict()
            for key, value in _prediction.items():
                prediction[key] = value.detach().cpu().numpy()
            predictions.append(prediction)
            # batches.append(batch)

    predictions = stacked_dict(predictions)
    # batches = stacked_dict(batches)
    return predictions  # , batches


def run_config(args, config, checkpoint):
    run_path = 'runs/{}'.format(config['run_id'])
    run_module = importlib.import_module(config['modules']['run']['name'])
    dataloader_module = importlib.import_module(config['modules']['dataloader']['name'])

    run = run_module.Run(config, args.cuda)
    run.load(checkpoint)

    if args.metadata is not None:
        sequences = dataloader_module.get_dataset_individually(
            base_directory=args.basedir,
            metadata_filename=args.metadata,
            split=args.foldname,
            input_context=config['modules']['dataloader']['input_context'],
            target_maxfilter=config['modules']['dataloader']['target_maxfilter'],
            audio_options=config['audio_options']
        )
    else:
        lc = config['modules']['dataloader']['args'][args.foldname]
        sequences = dataloader_module.get_dataset_individually(
            base_directory=lc['base_directory'],
            metadata_filename=lc['metadata_filename'],
            split=lc['split'],
            input_context=config['modules']['dataloader']['input_context'],
            target_maxfilter=config['modules']['dataloader']['target_maxfilter'],
            audio_options=config['audio_options']
        )

    print('len(sequences)', len(sequences))

    for si, sequence in enumerate(sequences):
        print(sequence.metadata['audio_filename'])
        predictions = predict_one_sequence(args, config, run, sequence)
        activations = np.stack([
            predictions['y_frames'],
            predictions['y_onsets'],
            predictions['y_offsets']
        ], axis=-1)

        predicted = dict(
            metadata=sequence.metadata,
            activations=activations
        )
        torch.save(predicted, os.path.join(args.outdir, 'predictions_{}.pkl'.format(si)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('foldname', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('--metadata', type=str)
    parser.add_argument('--basedir', type=str)
    parser.add_argument('--cuda', default=False, action='store_true')

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        print('outdir does not exist, creating...')
        os.makedirs(args.outdir)

    path_to_config, _ = os.path.split(args.checkpoint)
    configfile = os.path.join(path_to_config, 'config.pkl')
    config = torch.load(configfile)
    run_config(args, config, args.checkpoint)


if __name__ == '__main__':
    main()
