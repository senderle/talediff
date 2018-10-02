# cython: language_level=3

from libc.math cimport log, exp, isnan, isinf
import cython
import numpy
cimport numpy

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void zero(double[:, :] d2_buffer) nogil:
    cdef long d2_buffer_size = d2_buffer.shape[0]
    cdef long i, j
    for i in range(d2_buffer_size):
        for j in range(d2_buffer_size):
            d2_buffer[i, j] = 0

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void cy_train_chunk_combined(
        long[:] ends,
        long[:] indices,
        double[:] counts,
        double[:] totals,
        double[:] log_totals,
        double[:, :] embed_vecs,
        signed char[:, :] hash_vecs) nogil:

    # Memorably named temp variables.
    cdef double len_doc_recip = 0
    cdef double poly_pow = 0
    cdef double jac_i = 0
    cdef double counts_i = 0
    cdef double totals_i = 0
    cdef double hess_i_j = 0
    cdef double embed_dot = 0

    # Boundaries and index variables.
    cdef long n_docs = ends.shape[0]
    cdef long start = 0
    cdef long end = 0
    cdef long hess_size = 0
    cdef long embed_size = hash_vecs.shape[1]
    cdef long i, j, k, w_i, w_j, doc_ix

    # This is a bit inscrutable, because it uses two levels of
    # indirection to efficiently iterate over a ragged array of
    # arrays and perform a sparse dot product.

    # The boundaries of each array in the ragged array are defined
    # by `ends`. This iterates over each document in turn by
    # updating `end` and `start` appropriately, and then iterating
    # over the ragged arrays `indices`, `counts`, `totals`,
    # and `log_totals`. `indices` contains information about where
    # in the embedding and hash vector tables the given words are
    # located, which is why this can perform a sparse dot product;
    # it only updates the rows in `embed_vecs` for which `hess`
    # is nonzero.

    for doc_ix in range(n_docs):
        end = ends[doc_ix]
        hess_size = end - start

        # Calculate the sum of word counts for the given documents.
        # This will be used to calculate mean values later.
        len_doc_recip = 0
        for i in range(hess_size):
            len_doc_recip += counts[start + i]

        # Documents with only one word do us no good.
        if len_doc_recip < 2:
            continue

        len_doc_recip = 1 / len_doc_recip

        # Calculate the value of the pre-derivative polynomial
        # term (in log space). For a two-variable term, this is
        # just x ^ n * y ^ m. (Curious that a polynomial term
        # in log space is just the dot product of values and
        # exponents... I'm sure a mathematician would hit me
        # over the head with a ruler for not seeing immedaitely
        # why this should be so.)
        poly_pow = 0
        for i in range(hess_size):
            poly_pow = poly_pow + log(log_totals[start + i]) * counts[start + i]

        # Calculate the value of the hessian for the term, mostly
        # in log space. Then multiply it (dot product) by the
        # random hash vectors, as a low-budget form of dimension
        # reduction.
        for i in range(hess_size):
            w_i = indices[start + i]
            counts_i = counts[start + i]
            totals_i = totals[start + i]
            jac_i = (poly_pow +
                     log(counts_i) -
                     log(log_totals[start + i]))

            for j in range(hess_size):
                w_j = indices[start + j]

                # This turns x ^ n * y ^ m into n * x ^ (n - 1) *
                # m * y ^ (m - 1), implementing the power rule
                # for mixed partial derivatives. It happens in
                # log space.
                hess_i_j = exp(jac_i +
                               log(counts[start + j]) -
                               log(log_totals[start + j]))

                # This turns n * n * x ^ (n - 2) into n * (n - 1) *
                # x ^ (n - 2) along the diagonal *only*. This is
                # necessary because the diagonal contains second
                # partial derivatives of one variable instaed of
                # mixed partial derivatives of two variables. It
                # happens in linear space because (n - 1) will
                # often be zero, and a zero multiplication in log
                # space could be trouble. (0 == -inf in log space!)
                if i == j:
                    hess_i_j *= (counts_i - 1) / counts_i

                # This takes the geometric mean *and* arithmetic
                # mean w.r.t. doc length. It's a really weird
                # thing to do, but it "just works." I can't come
                # up with a theoretical justification, and it could
                # be that it simply works by pusing most values so
                # close to one that we might as well just be using
                # a plain old word-word coocurrence count matrix.
                # But I hope not!

                # A slightly more justified approach would calculate
                # a jacobian as well as a hessian, and use it as a
                # global normalizer after all the sparse hessians
                # and jacobians have been summed together. That is
                # implemented by the below function.

                # This also takes the arithmetic mean w.r.t total
                # word frequency.

                # Practially speaking, the effect of all of these
                # steps is to attenuate the absurd scaling effects
                # that, if left unaddressed, would cause massive
                # overflows -- while at the same time, preserving
                # some (lots?) of the structure that a straight
                # word-word cooccurrence count matrix would lose.

                # It's possible -- even likely -- that these steps
                # have probabilistic interpretations, or that this
                # reduces to something like a ppmi matrix of the
                # kind described by Goldberg and Levy (2014).
                hess_i_j = (hess_i_j ** len_doc_recip
                            * len_doc_recip / (totals_i + 1))

                # A bit sloppy: assume nan == 0
                if isnan(hess_i_j) or isinf(hess_i_j):
                    continue

                # Sparse dot product over random hash vectors.
                for k in range(embed_size):

                    embed_dot = (embed_vecs[w_i, k] +
                                 hess_i_j *
                                 hash_vecs[w_j, k])
                    embed_vecs[w_i, k] = embed_dot

        start = end

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void cy_train_chunk_jacobian(
        long[:] ends,
        long[:] indices,
        double[:] counts,
        double[:] totals,
        double[:] log_totals,
        double[:] jacobian,
        double[:, :] embed_vecs,
        signed char[:, :] hash_vecs) nogil:

    # Memorably named temp variables.
    cdef double len_doc_recip = 0
    cdef double poly_pow = 0
    cdef double jac_i = 0
    cdef double counts_i = 0
    cdef double totals_i = 0
    cdef double hess_i_j = 0
    cdef double embed_dot = 0

    # Boundaries and index variables.
    cdef long n_docs = ends.shape[0]
    cdef long start = 0
    cdef long end = 0
    cdef long hess_size = 0
    cdef long embed_size = hash_vecs.shape[1]
    cdef long i, j, k, w_i, w_j, doc_ix

    # This performs almost the same operation as the above function
    # `cy_train_chunk_combined`, but in addition to updating the
    # hessian matrix, it updates a jacobian vector, which can then
    # be used later, outside the tight loop -- as a normalization
    # constant, for example. See the notes to the above function
    # for details about the ragged array iteration being performed
    # here. In places where the two functions are the same, the
    # comments here are retained, but occasionally simplified.

    for doc_ix in range(n_docs):
        end = ends[doc_ix]
        hess_size = end - start

        # Calculate the sum of word counts for the given documents.
        # This will be used to calculate mean values later.
        len_doc_recip = 0
        for i in range(hess_size):
            len_doc_recip += counts[start + i]

        # Documents with only one word do us no good.
        if len_doc_recip < 2:
            continue

        len_doc_recip = 1 / len_doc_recip

        # Calculate the value of the pre-derivative polynomial
        # term (in log space). For a two-variable term, this is
        # just x ^ n * y ^ m.
        poly_pow = 0
        for i in range(hess_size):
            poly_pow = poly_pow + log(log_totals[start + i]) * counts[start + i]

        # Calculate the value of the hessian for the term, mostly
        # in log space. Then multiply it (dot product) by the
        # random hash vectors, as a low-budget form of dimension
        # reduction.
        for i in range(hess_size):
            # Precompute variables depending only on i
            w_i = indices[start + i]
            counts_i = counts[start + i]
            totals_i = totals[start + i]

            # Calculate the jacobian for word i
            jac_i = (poly_pow +
                     log(counts_i) -
                     log(log_totals[start + i]))
            jacobian[w_i] = (jacobian[w_i] +
                             exp(jac_i * len_doc_recip))
            for j in range(hess_size):
                # This turns x ^ n * y ^ m into n * x ^ (n - 1) *
                # m * y ^ (m - 1), implementing the power rule
                # for mixed partial derivatives. It happens in
                # log space.
                hess_i_j = exp(jac_i +
                               log(counts[start + j]) -
                               log(log_totals[start + j]))

                # This turns n * n * x ^ (n - 2) into n * (n - 1) *
                # x ^ (n - 2) along the diagonal *only*. This is
                # necessary because the diagonal contains second
                # partial derivatives of one variable instaed of
                # mixed partial derivatives of two variables. It
                # happens in linear space because (n - 1) will
                # often be zero, and a zero multiplication in log
                # space could be trouble. (0 == -inf in log space!)
                if i == j:
                    hess_i_j *= (counts_i - 1) / counts_i

                # Geometric mean w.r.t. doc length and arithmetic
                # mean w.r.t total word frequency.
                # hess_i_j = (hess_i_j ** len_doc_recip /
                #             (totals_i))
                # hess_i_j = hess_i_j / totals_i
                hess_i_j = hess_i_j ** len_doc_recip

                # A bit sloppy: assume nan == 0
                if isnan(hess_i_j) or isinf(hess_i_j):
                    continue

                # Sparse dot product over random hash vectors.
                w_j = indices[start + j]
                for k in range(embed_size):
                    embed_dot = (embed_vecs[w_i, k] +
                                 hess_i_j *
                                 hash_vecs[w_j, k])
                    embed_vecs[w_i, k] = embed_dot

        start = end

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void cy_train_chunk_configurable(
        long[:] ends,
        long[:] indices,
        double[:] counts,
        double[:] totals,
        double[:] log_totals,
        double[:] jacobian,
        double[:, :] embed_vecs,
        signed char[:, :] hash_vecs,
        int arithmetic_norm,
        int geometric_norm) nogil:

    # Memorably named temp variables.
    cdef double len_doc_recip = 0
    cdef double poly_pow = 0
    cdef double jac_i = 0
    cdef double counts_i = 0
    cdef double totals_i = 0
    cdef double hess_i_j = 0
    cdef double embed_dot = 0

    # Boundaries and index variables.
    cdef long n_docs = ends.shape[0]
    cdef long start = 0
    cdef long end = 0
    cdef long hess_size = 0
    cdef long embed_size = hash_vecs.shape[1]
    cdef long i, j, k, w_i, w_j, doc_ix

    # This performs almost the same operation as the above function
    # `cy_train_chunk_jacobian`, but provides several configurable
    # behaviors. Notes are omitted entirely; for documentation, see
    # comments to above functions.

    for doc_ix in range(n_docs):
        end = ends[doc_ix]
        hess_size = end - start

        len_doc_recip = 0
        for i in range(hess_size):
            len_doc_recip += counts[start + i]

        if len_doc_recip < 2:
            continue

        len_doc_recip = 1 / len_doc_recip

        poly_pow = 0
        for i in range(hess_size):
            poly_pow = poly_pow + log(log_totals[start + i]) * counts[start + i]

        for i in range(hess_size):
            w_i = indices[start + i]
            counts_i = counts[start + i]
            totals_i = totals[start + i]

            jac_i = (poly_pow +
                     log(counts_i) -
                     log(log_totals[start + i]))

            jacobian[w_i] = (jacobian[w_i] +
                             exp(jac_i * len_doc_recip))
            for j in range(hess_size):
                hess_i_j = exp(jac_i +
                               log(counts[start + j]) -
                               log(log_totals[start + j]))

                if i == j:
                    hess_i_j *= (counts_i - 1) / counts_i
                if geometric_norm > 0:
                    hess_i_j = hess_i_j ** len_doc_recip
                if arithmetic_norm > 0:
                    hess_i_j = hess_i_j / totals_i
                if isnan(hess_i_j) or isinf(hess_i_j):
                    continue

                w_j = indices[start + j]
                for k in range(embed_size):
                    embed_dot = (embed_vecs[w_i, k] +
                                 hess_i_j *
                                 hash_vecs[w_j, k])
                    embed_vecs[w_i, k] = embed_dot

        start = end

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void cy_train_chunk(
        long[:] ends,
        long[:] indices,
        double[:] counts,
        double[:] totals,
        double[:] log_totals,
        double[:, :] hess_buffer,
        double[:, :] embed_vecs,
        signed char[:, :] hash_vecs) nogil:
    cdef long start = 0
    cdef long end = 0
    cdef long doc_ix = 0
    cdef long n_docs = ends.shape[0]

    for doc_ix in range(n_docs):
        end = ends[doc_ix]

        cy_embed_hess(counts[start: end],
                      log_totals[start: end],
                      totals[start: end],
                      indices[start: end],
                      hess_buffer,
                      embed_vecs,
                      hash_vecs)
        start = end

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void cy_embed_hess(
        double[:] dc_vec,
        double[:] wc_vec,
        double[:] wc_vec_raw,
        long[:] index_vec,
        double[:, :] hess,
        double[:, :] embed,
        signed char[:, :] hash_vecs) nogil:
    cdef double len_doc_recip = 0
    cdef long hess_size = dc_vec.shape[0]
    cdef long embed_size = hash_vecs.shape[1]
    cdef long i, j, k, w_i, w_j
    cdef double hess_norm, result

    for i in range(hess_size):
        len_doc_recip += dc_vec[i]

    # Documents with only one word do us no good.
    if len_doc_recip < 2:
        return

    len_doc_recip = 1 / len_doc_recip

    cy_block_logspace_hessian(dc_vec, wc_vec, hess)
    for i in range(hess_size):
        for j in range(hess_size):
            hess_norm = hess[i, j] ** len_doc_recip
            hess_norm *= len_doc_recip
            hess_norm /= wc_vec_raw[i] + 1

            # A bit sloppy: assume nan == 0
            if isnan(hess_norm) or isinf(hess_norm):
                continue

            for k in range(embed_size):
                w_i = index_vec[i]
                w_j = index_vec[j]
                embed[w_i, k] = embed[w_i, k] + hess_norm * hash_vecs[w_j, k]

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void cy_block_logspace_hessian(
        double[:] pow_vector,
        double[:] x_vector,
        double[:, :] out) nogil:
    cdef long size = x_vector.shape[0]
    cdef long i, j
    cdef double jac_i = 0

    zero(out)

    cdef double p = 0
    for i in range(size):
        p = p + log(x_vector[i]) * pow_vector[i]

    for i in range(size):
        jac_i = p + log(pow_vector[i]) - log(x_vector[i])
        for j in range(size):
            out[i, j] = exp(jac_i +
                            log(pow_vector[j]) -
                            log(x_vector[j]))

    for i in range(size):
        out[i, i] = out[i, i] * (pow_vector[i] - 1) / (pow_vector[i])

def train_chunk_cy(docarray, hash_vectors):
    ends, indices, counts, totals, log_totals = docarray.features()
    embed_vectors = numpy.zeros(hash_vectors.shape, dtype=numpy.float64)

    cy_train_chunk_combined(ends, indices, counts, totals,
                            log_totals, embed_vectors, hash_vectors)

    nonzero_ix = embed_vectors.sum(axis=1) > 0
    return embed_vectors[nonzero_ix], nonzero_ix.nonzero()

def train_chunk_jacobian_cy(docarray, hash_vectors):
    ends, indices, counts, totals, log_totals = docarray.features()
    embed_vectors = numpy.zeros(hash_vectors.shape, dtype=numpy.float64)
    jacobian = numpy.zeros(hash_vectors.shape[0], dtype=numpy.float64)

    cy_train_chunk_jacobian(ends, indices, counts, totals,
                            log_totals, jacobian, embed_vectors, hash_vectors)

    nonzero_ix = embed_vectors.sum(axis=1) > 0
    return embed_vectors[nonzero_ix], jacobian[nonzero_ix], nonzero_ix.nonzero()

def train_chunk_configurable_cy(docarray, hash_vectors,
                                arithmetic_norm=False,
                                geometric_norm=False):
    ends, indices, counts, totals, log_totals = docarray.features()
    embed_vectors = numpy.zeros(hash_vectors.shape, dtype=numpy.float64)
    jacobian = numpy.zeros(hash_vectors.shape[0], dtype=numpy.float64)

    arithmetic_norm = int(bool(arithmetic_norm))
    geometric_norm = int(bool(geometric_norm))
    cy_train_chunk_configurable(ends, indices, counts, totals, log_totals,
                                jacobian, embed_vectors, hash_vectors,
                                arithmetic_norm, geometric_norm)

    nonzero_ix = embed_vectors.sum(axis=1) > 0
    return (embed_vectors[nonzero_ix],
            jacobian[nonzero_ix],
            nonzero_ix.nonzero())

# This can be regarded as a middle-way reference implementation
# that uses pure cython objects for inner loops (via the
# `cy_embed_hess` and `cy_block_logspace_hessian` functions),
# but that manages the outer loops using ordinary python objects.
# The workflow is the same as that implemented above in
# `cy_train_chunk_combined`; this is slower but easier to
# comprehend and maintain.

# The intermediate results from `cy_embed_hess` and
# `cy_block_logspace_hessian` can also be unit-tested
# separately. We can thus convince ourselves that this flow
# is correct with high confidence, and then use it to test
# the ...`_combined` functions above, which are fast, but
# terribly ugly and unpleasant to think about.
def train_chunk_py(docarray, hash_vectors):
    embed_vectors = numpy.zeros(hash_vectors.shape, dtype=numpy.float64)
    hess_buffer = numpy.zeros((50, 50), dtype=numpy.float64)

    for indices, counts, totals, log_totals in docarray:
        len_doc = counts.sum()
        if len_doc < 2:
            continue

        if len_doc > hess_buffer.shape[0]:
            newshape = (int(len_doc * 2), int(len_doc * 2))
            hess_buffer = numpy.zeros(newshape, dtype=numpy.float64)

        cy_embed_hess(counts, log_totals, totals, indices,
                      hess_buffer, embed_vectors, hash_vectors)

    nonzero_ix = embed_vectors.sum(axis=1) > 0
    return embed_vectors[nonzero_ix], nonzero_ix.nonzero()

def block_logspace_hessian(pow_vector, x_vector):
    cdef long size = x_vector.shape[0]
    out_arr = numpy.zeros((size, size), dtype=numpy.float64)
    cy_block_logspace_hessian(pow_vector, x_vector, out_arr)
    return out_arr
