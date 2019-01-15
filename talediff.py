import os
import re
import string
import argparse
import subprocess
import numpy
from itertools import islice

from docembed import DocArray
from docembed import Embedding
from docembed import test_hessians

# A heisenbug occurred here. `set` produced random orderings of
# punctuation characters. Some of those orderings produced regexes
# that did bizarre things. This only happened occasionally and when
# I attempted to recreate the bug, everything behaved as normal.
# Only when I saw that the characters in `_punct` were in a
# different order each time did I understand what I had done.
# In Python 3, `hash` was updated to use a more cryptographically
# secure hash function, and to seed it anew every time a new
# Python process is started. Thus bad orderings appear only
# occasionally and unpredictably.
_punct = '\\'.join(sorted(set(string.punctuation) - set('.')))
_punct += '\\“\\”\\—\\…'
_punct_rex = re.compile('[{}]'.format(_punct))
def load_and_split(txt):
    sents = _punct_rex.sub('', txt.lower()).split('.')
    sents = [s.split() for s in sents]
    return sents


_allpunct_rex = re.compile('[{}]'.format(_punct + '\\.'))
def load_and_make_windows(txt, window_size=15):
    sents = _allpunct_rex.sub('', txt.lower()).split()
    sents = [sents[i: i + window_size]
             for i in range(0, len(sents), window_size)]
    return sents

def random_window_gen(mean, std, block_size=1000):
    while True:
        for v in numpy.random.normal(mean, std, block_size):
            yield int(v)

def load_and_make_random_windows(txt, window_size=15, reps=1):
    words = _allpunct_rex.sub('', txt.lower()).split()
    for i in range(reps):
        start = 0
        for size in random_window_gen(window_size, window_size // 2):
            if size < 3:
                continue
            end = start + size
            yield words[start:end]

            start = end
            if start >= len(words):
                break


# Here, "annotation" means creating overlapping windows,
# where very course-grained word position information is
# retained by prepending an underscore to the words in
# one half of the sentence and leaving the words in the
# second half unchanged.
def load_and_annotate_windows(txt, window_size=15):
    split = _allpunct_rex.sub('', txt.lower()).split()
    sents = (split[i: i + window_size]
             for i in range(0, len(split), window_size // 2))
    annotated = []
    for s in sents:
        s_ = [w + '_' for w in s]
        s_head = s[:window_size // 2]
        s_tail = s[window_size // 2:]
        s_head_ = s_[:window_size // 2]
        s_tail_ = s_[window_size // 2:]
        annotated.append(s_head + s_tail_)
        annotated.append(s_head_ + s_tail)
    return annotated

def load(fn, window_size=15, annotate=True):
    with open(fn) as ip:
        txt = ip.read()
    if '.' in txt:
        return load_and_split(txt)
    elif annotate:
        return load_and_annotate_windows(txt, window_size)
    else:
        # return load_and_make_windows(txt, window_size)
        return load_and_make_random_windows(txt, window_size)

def group(it, n):
    it = iter(it)
    g = tuple(islice(it, n))
    while g:
        yield g
        g = tuple(islice(it, n))

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
        '-j',
        '--with-jacobian',
        action='store_true',
        default=False,
        help='Calculate the jacobian for use as a normalizing factor.'
    )
    parser.add_argument(
        '-r',
        '--arithmetic-norm',
        action='store_true',
        default=False,
        help='Use arithmetic normalization.'
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
        '-a',
        '--annotate',
        action='store_true',
        default=False,
        help='Add annotations to words indicating their position in '
        'sentences. This is currently experimental, and does not yet '
        'produce good results.'
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
        '-i',
        '--number-of-iterations',
        type=int,
        default=1,
        help='The number of times to resample the resulting vectors and '
        'retrain the embeddings. The first iteration produces purely random '
        'projections of the hessian; later iterations produce more organized '
        'projections. More than three or four iterations appear to produce '
        'overfitting behavior.'
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
        '-t',
        '--save-truncated-vectors',
        type=int,
        default=0,
        help='Perform dimension reduction using SVD and preserve this many '
        'dimensions. By default, no dimension reduction is performed.'
    )
    parser.add_argument(
        '--test-hessian',
        action='store_true',
        default=False,
        help='Rather than training a model, perform a battery of tests to the '
        'hessian-generating code.'
    )
    parser.add_argument(
        '--test-train-chunk',
        action='store_true',
        default=False,
        help='Rather than training a model, perform a battery of tests to the '
        'chunkwise training code.'
    )
    parser.add_argument(
        '-d',
        '--demo',
        action='store_true',
        default=False,
        help='Display demo output after every iteration.'
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
                 '1scalefree', 'unitscalefree'],
        type=str,
        help='Mode for selecting a point in expressivity space for evaluating '
        'the jacobian and hessian. Defaults to the one-point (1, 1, 1, ...). '
    )

    args = parser.parse_args()
    if not args.demo and not args.evaluate:
        args.demo = True

    if args.test_train_chunk:
        args.number_of_windows = 10000

    return args

def main(args):
    textdir = args.text_directory

    textfiles = sorted(os.path.join(textdir, f) for f in os.listdir(textdir))
    alldocs = (sent
               for fn in textfiles
               for sent in load(fn,
                                window_size=args.window_size,
                                annotate=args.annotate))
    if args.number_of_windows > 0:
        alldocs = islice(alldocs, args.number_of_windows)

    alldocs = DocArray(alldocs,
                       eval_mode=args.eval_mode,
                       flatten_counts=args.flatten_counts)
    emb = Embedding(alldocs,
                    multiplier=args.hash_vector_multiplier,
                    sparsifier=args.hash_vector_sparsifier)

    print('Creating a {}-dimension base embedding.'.format(emb.n_bits))

    if args.test_train_chunk:
        emb.train_test_chunk_sp()
    else:
        main_train(emb, args)

def main_train(emb, args):
    # possibly my dimension reduction woes are arising because we
    # must complete at least two iterations for the SVD to produce
    # useful results...
    n_iters = args.number_of_iterations
    for i in range(n_iters):
        print('Iteration {}...'.format(i))
        emb.train_multi(with_jacobian=args.with_jacobian,
                        arithmetic_norm=args.arithmetic_norm,
                        geometric_scaling=args.geometric_scaling)
        if args.demo:
            demo_out(emb, i)

    try:
        truncate = int(args.save_truncated_vectors)
        if truncate == 0:
            truncate = None
    except ValueError:
        truncate = None

    emb.save_vectors('vectors.txt',
                     mincount=args.save_mincount,
                     truncate=truncate,
                     include_annotated=False)
    emb.save_vocab('vocab.txt', mincount=args.save_mincount)
    if args.evaluate:
        subprocess.run(['python', 'eval/python/evaluate.py'])


if __name__ == '__main__':
    args = parse_args()
    if args.test_hessian:
        print("running hessian tests")
        test_hessians()
    else:
        main(args)
