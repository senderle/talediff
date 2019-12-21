import os
import argparse
import subprocess
from itertools import islice, chain

from docembed import DocArray
from docembed import Embedding

import util

def demo_out(emb, iteration, testword=None):
    print()
    print('**** Iteration {} ****'.format(iteration))

    print()
    if testword is not None:
        if testword not in emb.docarray.word_index:
            print('Test word does not appear frequently enough in the corpus')
        else:
            print('Euclidean distance to test word')
            print(emb.closest_words(testword, n_words=50,
                                    euclidean=True, mincount=1))

            print()
            print('Cosine similarity to test word')
            print(emb.closest_words(testword, n_words=50, mincount=1))

            print()
            print('Sample vectors')
            print()
            print(testword + ':')
            print(emb.get_vec(testword))

    print()
    print('Interpretable dimensions?')
    print()
    print(emb.interpret_dimension(0, n_words=20, mincount=500))
    print(emb.interpret_dimension(1, n_words=20, mincount=500))
    print(emb.interpret_dimension(2, n_words=20, mincount=500))

    print()
    print('Analogy tests')
    print()
    if all(w in emb.docarray.word_index for w in ['arm', 'kick', 'leg']):
        print()
        print('Goal: arm movement; euclidean distance.')
        print(emb.analogy(
            positive=['arm', 'kick'],
            negative=['leg'],
            n_words=50,
            euclidean=True,
            mincount=1
        ))

        print()
        print('Goal: arm movement; cosine similarity.')
        print(emb.analogy(
            positive=['arm', 'kick'],
            negative=['leg'],
            n_words=50,
            mincount=1
        ))
    else:
        print('Arm movement analogy words do not appear '
              'frequently enough in the corpus')

    if all(w in emb.docarray.word_index for w in ['fleet', 'general', 'army']):
        print()
        print('Goal: pilot, captain, or admiral; euclidean distance.')
        print(emb.analogy(
            positive=['fleet', 'general'],
            negative=['army'],
            n_words=50,
            euclidean=True,
            mincount=1
        ))

        print()
        print('Goal: pilot, captain, or admiral; cosine similarity.')
        print(emb.analogy(
            positive=['fleet', 'general'],
            negative=['army'],
            n_words=50,
            mincount=1
        ))
    else:
        print('Naval officer analogy words do not appear '
              'frequently enoguh in the corpus')

    if all(w in emb.docarray.word_index for w in ['rey', 'han', 'kylo']):
        print()
        print('Goal: Rey\'s father; euclidean similarity.')
        print(emb.analogy(
            positive=['rey', 'han'],
            negative=['kylo'],
            n_words=50,
            euclidean=True,
            mincount=1
        ))

        print()
        print('Goal: Rey\'s father; cosine similarity.')
        print(emb.analogy(
            positive=['rey', 'han'],
            negative=['kylo'],
            n_words=50,
            mincount=1
        ))

    print()

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
        '--arithmetic-norm',
        action='store_true',
        default=False,
        help='Use arithmetic normalization.'
    )
    parser.add_argument(
        '-C',
        '--cosine-norm',
        action='store_true',
        default=False,
        help='Normalize results using cosine distance after every batch.'
    )
    parser.add_argument(
        '-g',
        '--geometric-scaling',
        type=float,
        default=1,
        help='A geometric scaling factor. A value of 0 is equivalent to an '
        'ordinary power of the terms for a sentence, while a value of 1 is '
        'equivalent to the geometric mean of the terms. Since sentence '
        'length may vary, nonzero values smooth out variation. But they also '
        'have subtle effects on the way the hessian approximates '
        'nonlinearities caused by repeated terms. Defaults to 1.'
    )
    parser.add_argument(
        '-w',
        '--window-size',
        type=int,
        default=60,
        help='The width of the average context window (for input that '
        'cannot be parsed into sentences). Defaults to 60. The length of '
        'each window is sampled from a gaussian distribution with this as '
        'the mean. (For comparison, the default GloVe settings use a '
        'bidirectional 15-word window, effectively resulting in a 30-word '
        'window, and GloVe does not randomly vary the window size.'
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        type=int,
        default=20_000_000,
        help='The size of a single training batch. Default is '
        'twenty million words.'
    )
    parser.add_argument(
        '-r',
        '--window-sigma',
        type=float,
        default=0,
        help='The standard distribution of the window size distribution, '
        'as a fraction of the window size. Defaults to 0.5'
    )
    parser.add_argument(
        '-f',
        '--flatten-counts',
        action='store_true',
        default=False,
        help='Remove all repetitions, "flattening" the word counts for each '
        'sentence. This gives a more standard version of the coocurrence '
        'matrix.'
    )
    parser.add_argument(
        '-m',
        '--hash-vector-multiplier',
        type=int,
        default=5,
        help='A parameter that helps determine the length of the hash '
        'vectors, along with --hash-vector-sparsifier, below. We generate '
        'random binary projection vectors using the hash of the given '
        'word; hashes are 64-bits long, so if the sparsifier is 0, then '
        'a multiplier of 5 gives a 320-dimension vector. The total number '
        'of dimensions is equal to 64 * multiplier * sparsifier.'
    )
    parser.add_argument(
        '-s',
        '--hash-vector-sparsifier',
        type=int,
        default=-1,
        help='A parameter that "stretches" the hash vectors by padding them '
        'with evenly spaced zeros. This may help model interpretability. See '
        '--hash-vector-multiplier as well. A value less than zero indicates '
        'dense vectors of positive and negative values, instead of ones and '
        'zeros (the default behavior).'
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
        '-V',
        '--max-vocab',
        type=int,
        default=600000,
        help='The maximum number of terms to include in the vocabulary. Words '
        'are added on a "first-come-first-serve" basis until this number is '
        'reached. Defaults to 600,000.'
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
        '--test-hessian',
        action='store_true',
        default=False,
        help='Rather than training a model, perform a battery of tests to the '
        'hessian-generating code.'
    )
    parser.add_argument(
        '-e',
        '--evaluate',
        action='store_true',
        default=False,
        help='Execute the GloVe evaluation script on saved vectors.'
    )
    parser.add_argument(
        '-v',
        '--eval-mode',
        default='1',
        choices=['1', 'log', '1log', 'scalefree',
                 '1scalefree', 'unitscalefree', 'lognorm', 'wordnet'],
        type=str,
        help='Mode for selecting a point in expressivity space for evaluating '
        'the jacobian and hessian. Defaults to the one-point (1, 1, 1, ...). '
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
    emb = Embedding(DocArray(eval_mode=args.eval_mode,
                             flatten_counts=args.flatten_counts,
                             multiplier=args.hash_vector_multiplier,
                             sparsifier=args.hash_vector_sparsifier,
                             max_vocab=args.max_vocab))

    print('Creating a {}-dimension base embedding.'.format(emb.n_bits))

    n_words = 0

    batch_size = args.batch_size // args.window_size
    batches = batch_doc_iter(textdir,
                             args.window_size,
                             args.window_sigma,
                             args.number_of_windows,
                             batch_size)

    for batch_n, batch in enumerate(batches):
        print(' Batch {}...'.format(batch_n))

        emb.overwrite_docarray(batch)
        emb.train_multi(cosine_norm=args.cosine_norm,
                        arithmetic_norm=args.arithmetic_norm,
                        geometric_scaling=args.geometric_scaling)

        n_words += batch_size * args.window_size
        print(' {} tokens processed'.format(n_words))
        if args.save_interval > 0 and batch_n and not batch_n % args.save_interval:
            emb.save_vectors('vectors.txt',
                             mincount=args.save_mincount)
            emb.save_vocab('vocab.txt', mincount=args.save_mincount)
            if args.evaluate:
                subprocess.run(['python', 'eval/python/evaluate.py'])

    emb.save_vectors('vectors.txt',
                     mincount=args.save_mincount)
    emb.save_vocab('vocab.txt', mincount=args.save_mincount)
    if args.evaluate:
        subprocess.run(['python', 'eval/python/evaluate.py'])


if __name__ == '__main__':
    args = parse_args()
    if args.test_hessian:
        print("running hessian tests")
        util.test_hessians()
    else:
        main(args)
