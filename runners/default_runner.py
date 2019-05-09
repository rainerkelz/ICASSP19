import torch
import torch.optim.lr_scheduler as lr_scheduler
from utils import lr_scheduler_ext, stacked_dict

from torch import nn

import pickle
import numpy as np

from collections import defaultdict
import importlib
from utils import WrappedSummaryWriter
import time


def debug_gradients_tbx(logger, config, net, epoch):
    debug = config.get('debug', dict())

    if debug.get('weights', False):
        for name, param in net.named_parameters():
            logger.add_histogram('params/{}'.format(name), param.clone().detach().cpu().numpy(), epoch, bins='doane')  # bins='auto' causes tensorboard to implode occasionally

    if debug.get('gradients', False):
        for name, param in net.named_parameters():
            logger.add_histogram('grads/{}'.format(name), param.grad.clone().detach().cpu().numpy(), epoch, bins='doane')  # bins='auto' causes tensorboard to implode occasionally


def debug_gradients(config, bi, net):
    debug = config.get('mpl_debug', dict())
    if len(debug) == 0:
        return

    if bi % debug['step'] == 0:
        # this should be fine as a conditional import ...
        from utils import tensor_plot_helper, colorbar
        import matplotlib.pyplot as plt
        params = {
            'legend.fontsize': 'xx-small',
            'axes.labelsize': 'xx-small',
            'axes.titlesize': 'xx-small',
            'xtick.labelsize': 'xx-small',
            'ytick.labelsize': 'xx-small'
        }
        plt.rcParams.update(params)
        named_parameters = list(net.named_parameters())
        if debug['what'] == 'tensors':
            for name, parameter in named_parameters:
                weight = parameter.data.cpu().numpy()
                grad = parameter.grad.data.cpu().numpy()

                fig, axes = plt.subplots(ncols=2)
                fig.suptitle(name)
                axes[0].set_title('{} w'.format(name))
                axes[1].set_title('{} g'.format(name))

                im = tensor_plot_helper(fig, axes[0], weight, 'seismic', symm=True, preference='vertical')
                colorbar(im)

                im = tensor_plot_helper(fig, axes[1], grad, 'seismic', symm=True, preference='vertical')
                colorbar(im)
                plt.show()
        elif debug['what'] == 'histograms':
            fig, axes = plt.subplots(ncols=len(named_parameters), nrows=2)
            axes[0, 0].set_ylabel('w')
            axes[1, 0].set_ylabel('g')
            grad_magnitudes = []
            names = []
            for i, (name, parameter) in enumerate(named_parameters):
                names.append(name)
                weight = parameter.data.cpu().numpy()
                grad = parameter.grad.data.cpu().numpy()
                grad_magnitudes.append(np.sqrt(np.sum(grad ** 2)))

                axes[0, i].set_title(name)

                axes[0, i].hist(weight.flatten(), bins='auto', histtype='step', density=True)
                axes[1, i].hist(grad.flatten(), bins='auto', histtype='step', density=True)

            fig, ax = plt.subplots()
            ax.set_title('gradient magnitudes')
            lll = len(grad_magnitudes)
            xs = np.arange(lll)
            ax.vlines(xs, np.zeros(lll), grad_magnitudes)
            ax.set_xticks(xs)
            ax.set_xticklabels(names, rotation=90)
            plt.show()


def recursive_detach(din):
    dout = dict()
    for key, value in din.items():
        if isinstance(value, dict):
            dout[key] = recursive_detach(din[key])
        else:
            dout[key] = din[key].detach().cpu().numpy()
    return dout


class Run(object):
    def __init__(self, config, use_cuda=False, instantiate_net=True):
        self.config = config
        self.net = None

        if instantiate_net:
            self.instantiate_net(config)

        self.measures = dict(
            train=defaultdict(list),
            valid=defaultdict(list)
        )

        if use_cuda:
            print('using cuda')
            self.cuda()
        else:
            print('using cpu')
            self.cpu()

        self.epoch = -1

    def __getattr__(self, name):
        if name == 'logger':
            log_dir = 'runs/{}/tensorboard'.format(self.config['run_id'])
            self.logger = WrappedSummaryWriter(log_dir)
            return self.logger

    def cuda(self):
        self.net.cuda()
        self.CUDA = True

    def cpu(self):
        self.net.cpu()
        self.CUDA = False

    def instantiate_net(self, config):
        net_module = importlib.import_module(config['modules']['net']['name'])
        self.net = net_module.Net(self.config)

        data_parallel = self.config.get('data_parallel')
        print('using data parallel {}'.format(data_parallel))
        if data_parallel is not None:
            temp_lf = self.net.get_loss_function
            self.net = nn.DataParallel(self.net, **data_parallel)
            self.net.get_loss_function = temp_lf

        print(self.net)
        print('n_params', np.sum([np.prod(p.size()) for p in self.net.parameters()]))

        optimizer_class = getattr(torch.optim, config['optimizer']['name'])
        self.optimizer = optimizer_class(self.net.parameters(), **config['optimizer']['params'])
        if hasattr(lr_scheduler, config['scheduler']['name']):
            scheduler_class = getattr(lr_scheduler, config['scheduler']['name'])
            self.scheduler = scheduler_class(self.optimizer, **config['scheduler']['params'])
        elif hasattr(lr_scheduler_ext, config['scheduler']['name']):
            scheduler_class = getattr(lr_scheduler_ext, config['scheduler']['name'])
            self.scheduler = scheduler_class(self.optimizer, **config['scheduler']['params'])
        else:
            raise RuntimeError('unknown LR scheduler {}'.format(config['scheduler']))

    def set_dataloaders(self, dataloaders):
        self.dataloaders = dataloaders

    def advance(self):
        abort = self.train_one_epoch(self.dataloaders['train'])
        if not abort:
            self.evaluate(self.dataloaders['valid'], 'valid', checkpoint=True)
        return abort

    def test(self):
        self.evaluate(self.dataloaders['test'], 'test', checkpoint=False)

    def train_one_epoch(self, loader):
        self.epoch += 1
        print('epoch', self.epoch)
        self.net.train()
        train_loss_function = self.net.get_train_loss_function()
        smoothed_loss = 1.
        t_elapsed = 0
        n_batches = float(len(loader))
        count = 0
        total_count = 0
        abort = False
        for batch in loader:
            t_start = time.time()
            if self.CUDA:
                for key in batch.keys():
                    batch[key] = batch[key].cuda()

            # zero out gradients !
            self.optimizer.zero_grad()
            predictions = self.net.forward(batch)
            loss = train_loss_function(predictions, batch)
            loss.backward()
            self.optimizer.step()
            count += 1
            total_count += 1

            smoothed_loss = smoothed_loss * 0.9 + loss.cpu().item() * 0.1

            # bail if NaN or Inf is encountered
            if np.isnan(smoothed_loss) or np.isinf(smoothed_loss):
                print('encountered NaN/Inf in smoothed_loss "{}"'.format(smoothed_loss))
                abort = True
                break

            t_end = time.time()
            t_elapsed += (t_end - t_start)
            if t_elapsed > 60:
                batches_per_second = count / t_elapsed
                t_rest = ((n_batches - total_count) / batches_per_second) / 3600.
                print('bps {:4.2f} eta {:4.2f} [h]'.format(batches_per_second, t_rest))
                t_elapsed = 0
                count = 0
            debug_gradients(self.config, total_count, self.net)

        # visu ############################################################
        debug_gradients_tbx(self.logger, self.config, self.net, self.epoch)
        ###################################################################

        self.logger.add_scalar('train/loss', smoothed_loss, global_step=self.epoch)

        # always recorded, if present, to keep track of lr-scheduler
        for gi, param_group in enumerate(self.optimizer.param_groups):
            if 'lr' in param_group:
                self.logger.add_scalar(
                    'train/lr',
                    param_group['lr'],
                    global_step=self.epoch
                )
            if 'momentum' in param_group:
                self.logger.add_scalar(
                    'train/momentum',
                    param_group['momentum'],
                    global_step=self.epoch
                )

        return abort

    def find_learnrate(self, init_value=1e-8, final_value=10., exp_avg=0.98):
        data_loader = self.dataloaders['train']
        max_num = 4096
        num = min(max_num, len(data_loader) - 1)
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        train_loss_function = self.net.get_train_loss_function()
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        learning_rates = []
        for batch in data_loader:
            batch_num += 1
            print('flr batch_num', batch_num)
            if self.CUDA:
                for key in batch.keys():
                    batch[key] = batch[key].cuda()

            self.optimizer.zero_grad()
            prediction = self.net.forward(batch)
            loss = train_loss_function(prediction, batch)

            dloss = loss.detach().cpu().item()
            if np.isinf(dloss) or np.isnan(dloss):
                print('encountered "{}" in loss at batch "{}"'.format(dloss, batch_num))
                break

            # compute the smoothed loss
            avg_loss = exp_avg * avg_loss + (1 - exp_avg) * loss.item()
            smoothed_loss = avg_loss / (1 - exp_avg ** batch_num)
            # stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 10 * best_loss:
                print('loss exploding, aborting')
                break
            if batch_num > max_num:
                print('max number of batches reached')
                break
            # record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # store the values
            losses.append(smoothed_loss)
            learning_rates.append(lr)
            # do the SGD step
            loss.backward()
            self.optimizer.step()
            # update the lr for the next step
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.semilogx(learning_rates, losses)
        ax.set_xlabel('learning rate (log scale)')
        ax.set_ylabel('loss')
        plt.show()

    def evaluate_loader(self, loader):
        print('evaluating loader')
        self.net.eval()
        predictions = []
        batches = []
        with torch.no_grad():
            for cpu_batch in loader:
                batch = dict()
                if self.CUDA:
                    for key in cpu_batch.keys():
                        batch[key] = cpu_batch[key].cuda()
                else:
                    batch = cpu_batch

                gpu_prediction = self.net.forward(batch)
                cpu_prediction = recursive_detach(gpu_prediction)
                predictions.append(cpu_prediction)
                batches.append(cpu_batch)

        predictions = stacked_dict(predictions)
        batches = stacked_dict(batches)
        return predictions, batches

    def evaluate(self, loaders, name, checkpoint=True):
        # if we indeed get a list of loaders, run inference / evaluate each individually
        if isinstance(loaders, list):
            predictions = []
            batches = []
            for loader in loaders:
                # FIXME: this metadata fishing will not work with
                # datasets which does not have this field!
                loader_predictions, loader_batches = self.evaluate_loader(loader)
                predictions.append(dict(
                    metadata=loader.dataset.metadata,
                    predictions=loader_predictions
                ))
                batches.append(dict(
                    metadata=loader.dataset.metadata,
                    batches=loader_batches
                ))
        else:
            predictions, batches = self.evaluate_loader(loaders)

        checkpoint_filenames = self.net.evaluate_aggregate_checkpoint(
            name,
            predictions,
            batches,
            self.logger,
            self.epoch,
            scheduler=self.scheduler
        )

        if checkpoint and checkpoint_filenames is not None:
            if isinstance(checkpoint_filenames, list):
                for checkpoint_filename in checkpoint_filenames:
                    self.save(checkpoint_filename)
            else:
                self.save(checkpoint_filenames)

    def process(self, dataloader_module, infile, outfile):
        self.net.eval()
        loader = dataloader_module.get_loader_for_file(self.config, infile, 'SequentialSampler')
        predictions = []
        batches = []
        with torch.no_grad():
            for batch in loader:
                if self.CUDA:
                    for key in batch.keys():
                        batch[key] = batch[key].cuda()

                prediction = self.net.forward(batch)

                predictions.append(prediction)
                batches.append(batch)

        predictions = stacked_dict(predictions)
        batches = stacked_dict(batches)

        torch.save(
            dict(predictions=predictions, batches=batches),
            outfile,
            pickle_module=pickle,
            pickle_protocol=pickle.HIGHEST_PROTOCOL
        )

    def save(self, filename):
        state = dict()
        state['epoch'] = self.epoch
        state['net_state_dict'] = self.net.state_dict()
        state['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(state, filename, pickle_module=pickle, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        state = torch.load(filename, map_location=lambda storage, loc: storage)
        if self.instantiate_net:
            self.net.load_state_dict(state['net_state_dict']),
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.epoch = state['epoch']
