import sys
import os
import re
import string
import argparse
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
        return load_and_make_windows(txt, window_size)

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
        '-w',
        '--window-size',
        type=int,
        default=15,
        help='The width of the context window (for input that cannot be '
        'parsed into sentences).'
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
        '-m',
        '--hash-vector-multiplier',
        type=int,
        default=10,
        help='A parameter that determines the length of the output vectors. '
        'We generate random binary projection vectors using the hash of the '
        'given word; hashes are 64-bits long, so a multiplier of 10 gives a '
        '640-dimension vector.'
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
        default=2,
        help='The number of times to resample the resulting vectors and '
        'retrain the embeddings. The first iteration produces purely random '
        'projections of the hessian; later iterations produce more organized '
        'projections. More than three or four iterations appear to produce '
        'overfitting behavior. Defaults to 2.'
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
        nargs='?',
        type=int,
        default=0,
        const=50,
        help='Perform dimension reduction using SVD and preserve this many '
        'dimensions. If only the flag is passed, the number of dimensions '
        'defaults to 50. If the flag is not passed, no dimension reduction '
        'is performed.'
    )
    parser.add_argument(
        '-H',
        '--test-hessian',
        action='store_true',
        default=False,
        help='Rather than training a model, perform a battery of tests to the '
        'hessian-generating code.'
    )
    return parser.parse_args()

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

    alldocs = DocArray(alldocs)
    emb = Embedding(alldocs, multiplier=args.hash_vector_multiplier)

    # possibly my dimension reduction woes are arising because we
    # must complete at least two iterations for the SVD to produce
    # useful results...
    n_iters = args.number_of_iterations
    for i in range(n_iters):
        emb.train_multi(with_jacobian=args.with_jacobian)
        demo_out(emb, i)

        if i < n_iters - 1:
            emb.step_embedding()

    truncate = args.save_truncated_vectors
    truncate = truncate if truncate > 0 else None
    emb.save_vectors('vectors.txt',
                     mincount=args.save_mincount,
                     truncate=truncate,
                     include_annotated=False)
    emb.save_vocab('vocab.txt', mincount=args.save_mincount)


if __name__ == '__main__':
    args = parse_args()
    if args.test_hessian:
        print("running hessian tests")
        test_hessians()
    else:
        main(args)
