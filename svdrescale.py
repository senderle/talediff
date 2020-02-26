import os
import argparse
import numpy

from sklearn.decomposition import FastICA

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

def svd_rescale(vecs, alpha=0.5, truncate=None, cos_norm=True):
    U, s, V = numpy.linalg.svd(vecs, full_matrices=False)
    U_rescale = U @ numpy.diag(s ** alpha)
    if cos_norm:
        U_r_norm = (U_rescale * U_rescale).sum(axis=1) ** 0.5
        U_rescale /= U_r_norm[:, None]
    if truncate is not None:
        return U_rescale[:, :truncate]
    else:
        return U_rescale

def ica_rescale(vecs, alpha=1.0, truncate=None, cos_norm=True):
    if truncate is None:
        truncate = vecs.shape[1]
    ica = FastICA(n_components=truncate)
    signals = ica.fit_transform(vecs)  # The equivalent of eigenvectors.
    weights = ica.mixing_
    # assert numpy.allclose(vecs, signals @ weights.T + ica.mean_)
    print(signals.shape)
    print(weights.shape)

    # Rescaling step. Not sure this will have the same benefits for ICA
    # as for PCA.
    # Equivalence here not clear; this didn't work.
    # signals = signals @ numpy.diag(weights ** alpha)
    # print(signals.shape)

    # Cosine (constant length) normalization. Not sure this will have the
    # same beneits either.
    if cos_norm:
        signals_norm = (signals * signals).sum(axis=1) ** 0.5
        signals /= signals_norm[:, None]

    # Truncation is baked into the ICA algorithm, so skip that.
    return signals

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
        '--use-ica',
        action='store_true',
        default=False,
        help='Use ICA instead of PCA for decomposition step.'
    )
    parser.add_argument(
        '-a',
        '--alpha-rescaling-factor',
        type=float,
        default=0.6,
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

    if args.use_ica:
        U_rescale = ica_rescale(vecs,
                                args.alpha_rescaling_factor,
                                args.truncate_dimensions,
                                not args.disable_cosine_normalization)
    else:
        U_rescale = svd_rescale(vecs,
                                args.alpha_rescaling_factor,
                                args.truncate_dimensions,
                                not args.disable_cosine_normalization)

    save_vecs(args.output, words, U_rescale)
