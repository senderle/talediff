import os
import argparse
import subprocess
from itertools import islice, chain

from docembed import DocArray
from docembed import Embedding

import util

def parse_args():
    parser = argparse.ArgumentParser(
        description='A prototype word embedding model based on type-level '
        'differential operators.'
    )
    parser.add_argument(
        'text_directory',
        help='A directory containing text files for training a model.'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Show detailed information about vectors from each batch.'
    )
    parser.add_argument(
        '-d',
        '--dimension',
        type=int,
        default=300,
        help='The size of the hash vectors (and by extension, the output '
        'word vectors).'
    )
    parser.add_argument(
        '-p', 
        '--projection-distribution',
        default='binary',
        choices=['binary', 'gaussian'],
        type=str,
        help='Scheme for generating a random projection matrix. By default '
        'values from {-1, 1} are chosen according to a balanced binary '
        'distribution. In `gaussian` mode values are sampled from a normal '
        'distribution with a mean of 0 and a standard deviation of 1/3 '
        '(giving ~99%% of all values in the range of (-1, 1).'
    )
    parser.add_argument(
        '--projection-seed',
        default=0,
        type=int,
        help='Seed for the random projection hash function. Defaults to 0 to '
        'give consistent results across runs (with the same settings).'
    )
    parser.add_argument(
        '-v',
        '--ambiguity-vector',
        default='1',
        choices=['1', 'log', 'scalefree', 'wordnet', 'wordnetlog10'],
        type=str,
        help='Method for selecting a point in ambiguity space for evaluating '
        'the jacobian and hessian. Defaults to the one-point (1, 1, 1, ...). '
    )
    parser.add_argument(
        '-S', 
        '--ambiguity-scale',
        default=1.0 / 3,
        type=float,
        help='A scaling factor for the ambiguity vector.'
    )
    parser.add_argument(
        '-B', 
        '--ambiguity-base',
        default=1.0,
        type=float,
        help='The minimum value for the ambiguity vector.'
    )
    parser.add_argument(
        '-D',
        '--downsample-threshold',
        default=1.0,
        type=float,
        help='Words that occur with a probability greater than this '
        'number will be downsampled according to the formula 1 - (t / p). '
        'The default is 1.0, meaning that no words are downsampled. '
        'For a typical English-language corpus, this tends to reduce '
        'the frequency of roughly the top 10 / t '
        'words. For example, a threshold of 1e-4 will downsample '
        'about 1000 words.'
    )
    parser.add_argument(
        '-C',
        '--cosine-norm',
        action='store_true',
        default=False,
        help='Normalize results using cosine distance after every batch.'
    )
    parser.add_argument(
        '-w',
        '--window-size',
        type=int,
        default=50,
        help='The width of the average context window (for input that '
        'cannot be parsed into sentences). Defaults to 50. The length of '
        'each window is sampled from a gaussian distribution with this as '
        'the mean. (For comparison, the default GloVe settings use a '
        'bidirectional 15-word window, effectively resulting in a 30-word '
        'window, and GloVe does not randomly vary the window size.'
    )
    parser.add_argument(
        '-r',
        '--window-sigma',
        type=float,
        default=0,
        help='The standard distribution of the window size distribution, '
        'as a fraction of the window size. Defaults to 0, meaning that all '
        'windows are the same length.'
    )
    parser.add_argument(
        '-n',
        '--number-of-windows',
        type=int,
        default=0,
        help='The number of context windows to load. By default, all windows '
        'are loaded.'
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=50_000_000,
        help='The size of a single training batch. Default is '
        'fifty million words.'
    )
    parser.add_argument(
        '-c',
        '--save-mincount',
        type=int,
        default=5,
        help='The minimum number of times a word must appear in the corpus '
        'to be saved in the vector output file (used for the GloVe eval '
        'script). Defaults to 5, which is the default for the GloVe test '
        'using hutter8 data.'
    )
    parser.add_argument(
        '-M',
        '--max-vocab',
        type=int,
        default=600_000,
        help='The maximum number of terms to include in the vocabulary. Words '
        'are added on a "first-come-first-serve" basis until this number is '
        'reached. Defaults to 600,000.'
    )
    parser.add_argument(
        '-V',
        '--save-vocab',
        type=int,
        default=300_000,
        help='The maximum number of terms to include in the output. This is '
        'distinct from --max-vocab; given --max-vocab 600000 and '
        '--save-vocab 300000, the 300,000 least common words are simply '
        'thrown out. Defaults to 300,000.'
    )
    parser.add_argument(
        '-i',
        '--save-interval',
        type=int,
        default=5,
        help='The save frequency, measured in batches. Defaults to 5, meaning '
        'that after every five batches, a new version of vectors will be saved. '
        'If set to zero, vectors will only be saved at the end of the process.'
    )
    parser.add_argument(
        '-e',
        '--evaluate',
        action='store_true',
        default=False,
        help='Execute the GloVe evaluation script on saved vectors.'
    )
    parser.add_argument(
        '--test-hessian',
        action='store_true',
        default=False,
        help='Rather than training a model, perform a battery of tests to the '
        'hessian-generating code.'
    )

    args = parser.parse_args()

    return args

def doc_iter(textdir, window_size, window_sigma,
             n_windows):
    textfiles = sorted(os.path.join(textdir, f) for f in os.listdir(textdir))
    alldocs = (sent for fn in textfiles
               for sent in util.load(fn,
                                     window_size=window_size,
                                     window_sigma=window_sigma))
    if n_windows > 0:
        return islice(alldocs, n_windows)
    else:
        return alldocs

def batch_doc_iter(textdir, window_size, window_sigma,
                   n_windows, batchsize):
    docs = doc_iter(textdir, window_size, window_sigma, n_windows)
    end = object()
    while True:
        batch = islice(docs, batchsize)

        # Peek ahead to see if the batch is empty...
        next_item = next(batch, end)
        if next_item is end:
            return
        else:
            yield chain((next_item,), batch)

def main(args):
    textdir = args.text_directory
    docs = DocArray(ambiguity_vector=args.ambiguity_vector,
                    ambiguity_scale=args.ambiguity_scale,
                    ambiguity_base=args.ambiguity_base,
                    hash_dimension=args.dimension,
                    hash_distribution=args.projection_distribution,
                    hash_seed=args.projection_seed,
                    max_vocab=args.max_vocab,
                    downsample_threshold=args.downsample_threshold)
    emb = Embedding(docs, verbose=args.verbose)

    print('Creating a {}-dimension embedding.'.format(emb.n_bits))
    print()

    n_words = 0

    batch_size = args.batch_size // args.window_size
    batches = batch_doc_iter(textdir,
                             args.window_size,
                             args.window_sigma,
                             args.number_of_windows,
                             batch_size)

    for batch_n, batch in enumerate(batches, start=1):
        print(' Batch {}...'.format(batch_n))

        emb.overwrite_docarray(batch)
        emb.train_multi(cosine_norm=args.cosine_norm)

        n_words += batch_size * args.window_size
        print('   {} tokens processed'.format(n_words))
        print()
        if args.save_interval > 0 and not batch_n % args.save_interval:
            emb.save_vectors('vectors.txt',
                             mincount=args.save_mincount,
                             n_words=args.save_vocab)
            emb.save_vocab('vocab.txt', 
                           mincount=args.save_mincount,
                           n_words=args.save_vocab)
            if args.evaluate:
                subprocess.run(['python', 'eval/python/evaluate.py'])

    if args.evaluate:
        subprocess.run(['python', 'eval/python/evaluate.py'])

if __name__ == '__main__':
    args = parse_args()
    if args.test_hessian:
        print("running hessian tests")
        util.test_hessians()
    else:
        main(args)
