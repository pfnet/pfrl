import argparse
import os

import matplotlib
matplotlib.use('Agg')  # Needed to run without X-server
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', type=str, default='')
    parser.add_argument('--file', action='append', dest='files',
                        default=[], type=str,
                        help='specify paths of scores.txt')
    parser.add_argument('--label', action='append', dest='labels',
                        default=[], type=str,
                        help='specify labels for scores.txt files')
    args = parser.parse_args()

    assert len(args.files) > 0
    assert len(args.labels) == len(args.files)

    for fpath, label in zip(args.files, args.labels):
        if os.path.isdir(fpath):
            fpath = os.path.join(fpath, 'scores.txt')
        assert os.path.exists(fpath)
        scores = pd.read_csv(fpath, delimiter='\t')
        plt.plot(scores['steps'], scores['mean'], label=label)

    plt.xlabel('steps')
    plt.ylabel('score')
    plt.legend(loc='best')
    if args.title:
        plt.title(args.title)

    fig_fname = args.files[0] + args.title + '.png'
    plt.savefig(fig_fname)
    print('Saved a figure as {}'.format(fig_fname))

if __name__ == '__main__':
    main()
