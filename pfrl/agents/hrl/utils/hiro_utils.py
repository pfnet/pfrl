import torch
import numpy as np
import os
import csv
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Logger():
    def __init__(self, log_path):
        self.writer = SummaryWriter(log_path)

    def print(self, name, value, episode=-1, step=-1):
        string = "{} is {}".format(name, value)
        if episode > 0:
            print('Episode:{}, {}'.format(episode, string))
        if step > 0:
            print('Step:{}, {}'.format(step, string))

    def write(self, name, value, index):
        self.writer.add_scalar(name, value, index)


def var(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def get_tensor(z):
    if len(z.shape) == 1:
        return var(torch.FloatTensor(z.copy())).unsqueeze(0)
    else:
        return var(torch.FloatTensor(z.copy()))


def _is_update(episode, freq, ignore=0, rem=0):
    if episode != ignore and episode % freq == rem:
        return True
    return False


def record_experience_to_csv(args, experiment_name, csv_name='experiments.csv'):
    # append DATE_TIME to dict
    d = vars(args)
    d['date'] = experiment_name

    if os.path.exists(csv_name):
        # Save Dictionary to a csv
        with open(csv_name, 'a') as f:
            w = csv.DictWriter(f, list(d.keys()))
            w.writerow(d)
    else:
        # Save Dictionary to a csv
        with open(csv_name, 'w') as f:
            w = csv.DictWriter(f, list(d.keys()))
            w.writeheader()
            w.writerow(d)


def listdirs(directory):
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
