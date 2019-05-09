from tensorboardX.summary import Summary, _clean_tag, make_np
from tensorboardX.src.summary_pb2 import HistogramProto
from tensorboardX import SummaryWriter
import numpy as np
import torch
import os


def histogram(name, values, bins, collections=None):
    name = _clean_tag(name)
    values = make_np(values)
    hist = make_histogram(values.astype(float), bins)
    return Summary(value=[Summary.Value(tag=name, histo=hist)])


# copied over, b/c original histogram code tried to be too fancy,
# mucked up the number of bins, and produced crappy histograms ...
def make_histogram(values, bins):
    values = values.reshape(-1)
    counts, limits = np.histogram(values, bins=bins)
    limits = limits[1:]

    # safety net for 1-bin hists ...
    if len(counts) == 1:
        counts, limits = np.histogram(values, bins=3)
        limits = limits[1:]

    sum_sq = values.dot(values)
    return HistogramProto(min=values.min(),
                          max=values.max(),
                          num=len(values),
                          sum=values.sum(),
                          sum_squares=sum_sq,
                          bucket_limit=limits,
                          bucket=counts)


def counter(name, counter_dict):
    name = _clean_tag(name)
    hist = make_histogram_from_counter(counter_dict)
    return Summary(value=[Summary.Value(tag=name, histo=hist)])


def make_histogram_from_counter(counter_dict):
    values = np.array(list(counter_dict.keys()))
    counts = np.array(list(counter_dict.values()))
    sum_sq = values.dot(values)
    return HistogramProto(min=values.min(),
                          max=values.max(),
                          num=len(values),
                          sum=values.sum(),
                          sum_squares=sum_sq,
                          bucket_limit=values,
                          bucket=counts)


class WrappedSummaryWriter(SummaryWriter):
    def __init__(self, log_dir=None, comment='', show_mpl=False):
        super().__init__(log_dir, comment)
        self.__show_mpl = show_mpl

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None):
        self.file_writer.add_summary(histogram(tag, values, bins), global_step, walltime)

    def add_counter(self, tag, counter_dict, global_step=None, walltime=None):
        self.file_writer.add_summary(counter(tag, counter_dict), global_step, walltime)

    def add_mpl_plot(self, tag, fig, global_step):
        if self.__show_mpl:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            from PIL import Image
            import io
            fig.canvas.draw()

            pil_image = Image.frombytes(
                'RGB',
                fig.canvas.get_width_height(),
                fig.canvas.tostring_rgb()
            )

            output = io.BytesIO()
            pil_image.save(output, format='PNG')
            pil_image_string = output.getvalue()
            output.close()
            img_summary = Summary.Image(
                height=pil_image.height,
                width=pil_image.width,
                colorspace=3,
                encoded_image_string=pil_image_string
            )
            summary = Summary(value=[Summary.Value(tag=_clean_tag(tag), image=img_summary)])
            self.file_writer.add_summary(summary, global_step=global_step)

    def add_table(self, tag, table, global_step):
        self.add_text(tag, '\n-|-\n'.join(['|'.join(row) for row in table]), global_step)

    def add_object(self, tag, obj, global_step):
        log_dir = self.file_writer.get_logdir()
        path_to_file = os.path.join(log_dir, tag)
        # make path to file exists
        if os.path.exists(path_to_file):
            # if it exists, and is not a directory, throw an exception
            if os.path.isfile(path_to_file):
                raise RuntimeError('path which should be a directory is a file instead')
        else:
            os.makedirs(path_to_file)

        full_file_path = os.path.join(path_to_file, 'step_{}.pkl'.format(global_step))
        torch.save(obj, full_file_path)
