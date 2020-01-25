import os
import argparse
import numpy

def load_vecs(path):
    with open(path) as ip:
        rows = [line.split() for line in ip]
    words = [r[0] for r in rows]
    vecs = numpy.array([list(map(float, r[1:])) for r in rows])
    return words, vecs

def save_vecs(path, words, vecs):
    vecs = vecs.astype('float16')
    with open(path, 'w') as op:
        for w, vec in zip(words, vecs):
            op.write('{} {}\n'.format(
                w, ' '.join(str(v) for v in vec)
            ))

def get_args():
    parser = argparse.ArgumentParser(
        description='A word vector SVD rescaler.'
    )

    parser.add_argument(
        'vectors',
        help='The path to a text-based vector file',
        type=str
    )
    parser.add_argument(
        'output',
        help='Save rescaled vectors to this path.',
        type=str
    )
    parser.add_argument(
        '-a',
        '--alpha-rescaling-factor',
        type=float,
        default=0.5,
        help='The rescaling parameter.'
    )
    parser.add_argument(
        '-t',
        '--truncate-dimensions',
        type=int,
        default=0,
        help='Preserve only this many dimensions.'
    )
    parser.add_argument(
        '--disable-cosine-normalization',
        action='store_true',
        default=False,
        help='Disable L2-normalization of the output.'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    words, vecs = load_vecs(args.vectors)
    if args.truncate_dimensions <= 0:
        args.truncate_dimensions = vecs.shape[1]

    U, s, V = numpy.linalg.svd(vecs, full_matrices=False)
    U_rescale = U @ numpy.diag(s ** args.alpha_rescaling_factor)
    if not args.disable_cosine_normalization:
        U_r_norm = (U_rescale * U_rescale).sum(axis=1) ** 0.5
        U_rescale /= U_r_norm[:, None]
    save_vecs(args.output, words, U_rescale[:, :args.truncate_dimensions])
