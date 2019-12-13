import re
import string
import numpy
import random

from sklearn.linear_model import SGDClassifier

from itertools import islice

from sparsehess import block_logspace_hessian as block_logspace_hessian_sp
from sparsehess import train_chunk_vanilla_full
from sparsehess import train_chunk_configurable_scaling

# Cause pyflakes to ignore unused imports; these are used in other modules
# via this one. No other module should directly refer to the `sparsehess`
# module.

_NO_WARN = (train_chunk_configurable_scaling, train_chunk_vanlla_full)

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
    sents = _allpunct_rex.sub(' ', txt.lower()).split()
    sents = [sents[i: i + window_size]
             for i in range(0, len(sents), window_size)]
    return sents

def random_window_gen(mean, std, block_size=1000):
    while True:
        for v in numpy.random.normal(mean, std, block_size):
            yield int(v)

def load_and_make_random_windows(txt, window_size=15,
                                 window_sigma=0.5, reps=1):
    words = _allpunct_rex.sub(' ', txt.lower()).split()
    for i in range(reps):
        start = 0
        for size in random_window_gen(window_size,
                                      int(window_size * window_sigma)):
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

def load(fn, window_size=15, window_sigma=0.5, annotate=False):
    with open(fn) as ip:
        txt = ip.read()

    # if '.' in txt:
    #     return load_and_split(txt)
    if annotate:
        return load_and_annotate_windows(txt, window_size)
    else:
        # return load_and_make_windows(txt, window_size)
        return load_and_make_random_windows(txt, window_size, window_sigma)

def group(it, n):
    it = iter(it)
    g = tuple(islice(it, n))
    while g:
        yield g
        g = tuple(islice(it, n))

def sparsify_rows(matrix, iters=1):
    for i in range(iters):
        left = matrix[:, 0::2]
        right = matrix[:, 1::2]

        rows, cols = matrix.shape
        cols *= 2

        out = numpy.zeros((rows, cols), dtype=numpy.uint8)
        out[:, 0::4] = left * (1 - right)
        out[:, 1::4] = (1 - left) * right
        out[:, 2::4] = (1 - left) * (1 - right)
        out[:, 3::4] = left * right

        matrix = out

    return matrix

# This is my quick-and-dirty implementation of what Ben Schmidt calls
# "Stable Random Projection." I'm not sure it's totally correct, but
# the idea is from him.
def srp_matrix(words, multiplier, sparsifier=-1):
    hashes = [
        list(map(hash, ['{}_{}'.format(w, i) for i in range(multiplier)]))
        for w in words
    ]

    # Given a `multipier` value of 5, `hashes` is really a Vx5
    # array of 8-byte integers, where V is the vocabulary size.

    hash_arr = numpy.array(hashes, dtype=numpy.int64)

    # But we could also think of it as an array of bytes,
    # where every word is represented by 40 bytes...

    hash_arr = hash_arr.view(dtype=numpy.uint8)

    # ...or even as an array of bits, where every word is represented
    # by 320 bits...

    hash_arr = numpy.unpackbits(hash_arr.ravel()).reshape(-1, 64 * multiplier)

    if sparsifier >= 0:
        # ...or even as an array of bits, where every word is represented
        # by 640 bits, where pairs of bits are mapped to four bits with one
        # positive value, which ensures greater sparsity, which is what
        # sparsify_rows does if `sparsifier` is greater than 1.
        out = sparsify_rows(hash_arr, sparsifier).astype(numpy.float64)
    else:
        out = hash_arr.astype(numpy.float64) * 2 - 1

    return out

def resample_vectors(vecs):
    new_vecs = numpy.empty(vecs.shape, dtype=vecs.dtype)
    for i, v in enumerate(vecs):
        bins = numpy.zeros(v.shape[0] + 1)
        bins[1:] = v.cumsum()
        binsample = numpy.random.random(len(bins) * 10) * bins.max()
        bincount = numpy.histogram(binsample, bins)[0]
        new_vecs[i, :] = bincount > numpy.median(bincount)
    return numpy.asarray(new_vecs, dtype=numpy.uint8)

def resample_vectors_median(vecs):
    new_vecs = vecs.copy()
    new_vecs_median = numpy.median(new_vecs, axis=1)[:, None]
    new_vecs_std = new_vecs.std(axis=1)[:, None]
    new_vecs[:] = new_vecs > (new_vecs_median + new_vecs_std * 1.5)
    return numpy.asarray(new_vecs, dtype=numpy.uint8)

def word_doc_matrix(words, documents):
    index = {w: i for i, w in enumerate(words)}
    mat = numpy.zeros((len(words), len(documents)), dtype=numpy.float64)
    for j, doc in enumerate(documents):
        doc_ix = [index[w] for w in doc]
        ct = numpy.bincount(doc_ix)
        mat[:len(ct), j] = ct
    return mat

def cosine_sim(a, b):
    norm = ((a @ a) * (b @ b)) ** 0.5
    return (a @ b) / norm if norm != 0 else 0

def cosine_sim_to(word, vec_dict):
    if isinstance(word, str):
        vec = vec_dict[word]
    else:
        vec = word

    def dist(w):
        return cosine_sim(vec_dict[w], vec)
    return dist

def euclidean_dist(a, b):
    d = a - b
    return (d @ d) ** 0.5

def euclidean_dist_to(word, vec_dict):
    if isinstance(word, str):
        vec = vec_dict[word]
    else:
        vec = word

    def dist(w):
        return euclidean_dist(vec_dict[w], vec)
    return dist

def select_vectors(text_samples, embedding, n):
    # This function will do the following:
    #     1) Pick a word in the middle of each sample, for around 1000 samples.
    #     2) Build a set of one-hot vectors for the chosen words for use as
    #        the prediction targets.
    #     3) Use the vectors for the preceding n and trailing n words as
    #        the features.
    #     4) Train a logistic regression model to predct the output with L1
    #        regularization.
    #     5) Train multiple such models with different regularization params.
    #     6) Calculate the coreelation between each dimension and the L1 param.
    #     7) Throw out the k vectors that are the most correlated with the
    #        L1 param (assuming a higher param = *stronger* regularization i.e.
    #        sparser models). Or probably, should just sort them.

    targets = []
    features = []
    vectors = embedding.embed_vectors
    wix = embedding.docarray.word_index
    for s in text_samples:
        if len(s) <= n * 2:
            continue

        if not all(w in wix for w in s):
            continue

        mid = len(s) // 2
        pos_feat = [vectors[wix[w]] for w in s[mid - n:mid]]
        pos_feat.extend(vectors[wix[w]] for w in s[mid + 1:mid + n])
        neg_feat = pos_feat[:]

        pos_feat.append(vectors[wix[s[mid]]])
        neg_feat.append(random.choice(vectors))

        features.append(numpy.array(pos_feat).reshape(-1))
        targets.append(1.0)
        features.append(numpy.array(neg_feat).reshape(-1))
        targets.append(0.0)

    features = numpy.array(features)
    targets = numpy.array(targets)
    shuffle = numpy.random.permutation(len(targets))

    feature_weights = []
    # penalties = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    penalties = [0.001, 0.003]
    for l1 in penalties:
        model = SGDClassifier(loss='log',
                              penalty='l1',
                              alpha=l1,
                              max_iter=100,
                              tol=1e-3)
        model.fit(X=features[shuffle], y=targets[shuffle])
        feature_weights.append(model.coef_)

    print(numpy.array(feature_weights).shape)


def train_chunk(docarray, hash_vectors):
    embed_vectors = numpy.zeros(hash_vectors.shape)
    for indices, counts, totals, log_totals in docarray:
        # Documents with only one word do us no good.
        if counts.sum() < 2:
            continue

        embed = embed_hess(counts,
                           log_totals,
                           totals,
                           hash_vectors[indices, :])

        # Update embeddings for the given words:
        embed_vectors[indices, :] += embed
    return embed_vectors

def embed_hess(dc_vec, wc_vec, wc_vec_raw, hv):
    # Create hessian...
    hess = block_logspace_hessian_sp(dc_vec, wc_vec)
    len_doc = dc_vec.sum()

    # Take geometric mean w.r.t sentence length, the
    # arithmetic mean w.r.t. sentence length, and the
    # arithmeic mean w.r.t word token count. I don't
    # know how to justify these moves theoretically.
    # My best shot is that these "whiten" the hessian,
    # causing it to approximate something more ortho-
    # normal. They certainly lead to better results
    # in practice, because they ensure that longer
    # sentences don't get more weight than shorter
    # ones and that common words don't get more
    # weight than rare ones.
    hess **= 1 / len_doc
    hess /= len_doc
    hess /= (wc_vec_raw[:, None] + 1)

    # This corresponds to the tests described as
    # "a bit sloppy" in sparsehess.pyx. There's a
    # chance these have a significant negative
    # effect on dimension reduction; more testing
    # is needed to verify that this isn't an
    # issue.
    hess[~numpy.isfinite(hess)] = 0

    # Project hessian into hashed vector space...
    return hess @ hv

def zero_if_nan(arr_2d, msg=None):
    if numpy.isnan(arr_2d).any():
        if msg is not None:
            print(msg)
        return numpy.nan_to_num(arr_2d)
    else:
        return arr_2d

def block_hessian(pow_vector, x_vector):
    """
    Take a single polynomial term and calculate the hessian for that term
    at a given point. The term is represented as a sequence of powers, and
    the point is represented as a sequence of values corresponding to each
    given power's dimension. For example, this will calculate the hessian
    matrix of the term x ^ 2 * y * z at the point (x, y, z) = (2, 3, 3):

        block_hessian([2, 1, 1], [2, 3, 3])

    For sparse polynomials of many variables, this allows us to calculate
    the hessian efficiently, using only terms with nonzero weights, and
    calculating term values using only those dimensions with nonzero
    powers. The resulting blocks can be summed together to produce a
    NxN hessian for very large N.

    For additional efficiency, this calculates the hessian in the
    following form:

        H(P, X, i, j) = P * M * K
            P = Pi[k](X_k ^ P_k)
            M = P_i * P_j / (X_i * X_j)
            K = (P_i - kdel(i, j)) / P_i

    Here, Pi is the usual product operator, kdel is the Kronecker delta,
    and i and j are row and column indices. The P term can be calcluated
    just once and reused for all values of i and j. The M term can be
    calculated using efficient matrix operations. And the K term is equal
    to 1 except for values along the diagonal, and so can be applied just
    to the diagonal values of the output of the M term.

    To see that this calculates the hessian matrix, observe that for
    any given i not equal to j, it evaluates to:

        P_i * X_i ^ P_i / X-i * P_j * X_j ^ P_j / X_j
            * Pi[k != i, k !=j](X_k ^ P_k)

    The first part of which simplifies to the power rule for partial
    derivatives over two different dimensions:

        P_i * X_i ^ (P_i - 1) * P_j * X_j ^ (P_j - 1) * ...

    When i is equal to j, it evaluates to:

        P_i * (P_i - 1) * X_i ^ P_i / (X_i * X_i)
            * Pi[k != i](X_k ^ P_k)

    The first part of which simplifies to the power rule for second
    partial derivatives:

        P_i * (P_i - 1) * X_i ^ (P_i - 2) ...

    The resulting matrix is thus the matrix of all combinations of
    all second derivatives of the input polynomial.
    """

    P = numpy.product(x_vector ** pow_vector)
    M_num = pow_vector[:, None] * pow_vector[None, :]
    M_den = x_vector[:, None] * x_vector[None, :]
    K = (pow_vector - 1) / pow_vector

    P *= M_num / M_den
    P[numpy.diag_indices_from(P)] *= K

    return zero_if_nan(P, "block_hessian: nan result in hessian")

def block_logspace_hessian(pow_vector, x_vector):
    """
    Identical to `block_hessian` but calculated in log space.
    """

    x_vector_log = numpy.log(x_vector)
    pow_vector_log = numpy.log(pow_vector)

    M = pow_vector_log[:, None] + pow_vector_log[None, :]
    M -= x_vector_log[:, None]
    M -= x_vector_log[None, :]

    P = (x_vector_log * pow_vector).sum() + M
    P = numpy.exp(P)

    # The zero mutliply here is cleaner in linear space.
    K = (pow_vector - 1) / pow_vector
    P[numpy.diag_indices_from(P)] *= K

    return zero_if_nan(P, "block_logspace_hessian: nan result in hessian")

def estimated_hessian(ps, xs, delta=1e-4):
    def poly_eval(ps, xs):
        return numpy.prod(xs ** ps)

    def est_at(dim, dim2=None):
        if dim2 is None:
            p_diff = xs.copy()
            n_diff = xs.copy()
            p_diff[dim] += delta
            n_diff[dim] -= delta

            out = poly_eval(ps, p_diff)
            out -= 2 * poly_eval(ps, xs)
            out += poly_eval(ps, n_diff)
            out /= delta * delta
            return out
        else:
            dim1 = dim
            p_p_diff = xs.copy()
            p_n_diff = xs.copy()
            n_p_diff = xs.copy()
            n_n_diff = xs.copy()

            p_p_diff[dim1] += delta
            p_p_diff[dim2] += delta
            p_n_diff[dim1] += delta
            p_n_diff[dim2] -= delta
            n_p_diff[dim1] -= delta
            n_p_diff[dim2] += delta
            n_n_diff[dim1] -= delta
            n_n_diff[dim2] -= delta

            out = poly_eval(ps, p_p_diff)
            out += poly_eval(ps, n_n_diff)
            out -= poly_eval(ps, p_n_diff)
            out -= poly_eval(ps, n_p_diff)
            out /= delta * delta * 4
            return out

    result = numpy.zeros((len(xs), len(xs)), dtype=xs.dtype)
    for i in range(len(xs)):
        for j in range(len(xs)):
            if i == j:
                result[i, i] = est_at(i)
            else:
                result[i, j] = est_at(i, j)
    return result

def test_hessians():
    n_vars = 20
    n_tests = 100
    PS = numpy.random.randint(1, 5, size=(n_tests, n_vars))
    PS = numpy.asarray(PS, dtype=numpy.float64)
    XS = numpy.random.random((n_tests, n_vars)) * 3
    delta = 1e-6

    failed = 0
    for ps, xs in zip(PS, XS):
        bh_result = block_hessian(ps, xs)
        blh_result = block_logspace_hessian(ps, xs)
        blhs_result = block_logspace_hessian_sp(ps, xs)
        est_result = estimated_hessian(ps, xs)

        test_versions = ((bh_result, "block_hessian"),
                         (blh_result, "block_logspace_hessian"),
                         (blhs_result, "block_logspace_hessian_sp"))

        for res, res_name in test_versions:
            try:
                assert numpy.allclose(est_result, res)
            except AssertionError:
                err = est_result - res
                err_abs = numpy.abs(err)
                abs_err_prop = err_abs.sum().sum() / est_result.sum().sum()

                est_result_nz = est_result.copy()
                est_result_nz[est_result == 0] = 1
                err_prop_mat = err_abs / est_result_nz
                err_max = err_prop_mat.max()
                err_median = numpy.median(err_prop_mat)
                worst_errors = (err_abs / est_result_nz) > (abs_err_prop * 5)
                worst_errors[est_result == 0] = False
                if abs_err_prop > (delta * 100):
                    print()
                    print("Hessian test failed for {}...".format(res_name))
                    print("Absolute proportional error: ", abs_err_prop)
                    print("Max error: ", err_max)
                    print("Median error: ", err_median)
                    if worst_errors.any():
                        loc = worst_errors.nonzero()
                        table = zip(est_result[worst_errors],
                                    res[worst_errors],
                                    loc[0],
                                    loc[1])
                        print("Worst error values -- ")
                        print("      "
                              "  Estimated value   "
                              "  Calculated value  "
                              "  Difference        "
                              "  at Row, Col       ")
                        table_fmt = (
                            "  {: 18.10f}"
                            "  {: 18.10f}"
                            "  {: 18.10f}"
                            "        {}")
                        for est, cal, r, c in table:
                            print(table_fmt.format(
                                est, cal, abs(est - cal), str((r, c))
                            ))
                    failed += 1

    if failed:
        print()
        print("{}/{} tests failed.".format(failed,
                                           n_tests * len(test_versions)))
        print()
        print("This probably doesn't indicate a major problem unless the")
        print("absolute proportional error is higher than 0.05, or is higher")
        print("than 0.001 for more than twenty or thirty tests.")
    else:
        print("All hessian tests passed!")
