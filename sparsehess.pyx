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
        double[:] weights,
        double[:, :] rand_hess,
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
    # and `weights`. `indices` contains information about where
    # in the embedding and hash vector tables the given words are
    # located, which is why this can perform a sparse dot product;
    # it only updates the rows in `rand_hess` for which `hess`
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
            poly_pow = poly_pow + log(weights[start + i]) * counts[start + i]

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
                     log(weights[start + i]))

            for j in range(hess_size):
                w_j = indices[start + j]

                # This turns x ^ n * y ^ m into n * x ^ (n - 1) *
                # m * y ^ (m - 1), implementing the power rule
                # for mixed partial derivatives. It happens in
                # log space.
                hess_i_j = exp(jac_i +
                               log(counts[start + j]) -
                               log(weights[start + j]))

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

                    embed_dot = (rand_hess[w_i, k] +
                                 hess_i_j *
                                 hash_vecs[w_j, k])
                    rand_hess[w_i, k] = embed_dot

        start = end

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void cy_train_chunk_configurable_scaling(
        long[:] ends,
        long[:] indices,
        double[:] counts,
        double[:] totals,
        double[:] weights,
        double[:] jacobian,
        double[:, :] rand_hess,
        double[:, :] hash_vecs,
        double geometric_scaling,
        int arithmetic_norm) nogil:

    # Memorably named temp variables.
    cdef double len_doc_recip = 0
    cdef double poly_pow = 0
    cdef double jac_i = 0
    cdef double jac_i_norm = 0
    cdef double counts_i = 0
    cdef double totals_i = 0
    cdef double weights_i = 0
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
    # `cy_train_chunk_combined`, but provides several configurable
    # behaviors. Most notes are omitted entirely; for documentation, see
    # above comments.

    for doc_ix in range(n_docs):
        end = ends[doc_ix]
        hess_size = end - start

        len_doc_recip = 0
        for i in range(hess_size):
            len_doc_recip += counts[start + i]

        if len_doc_recip < 2:
            continue

        len_doc_recip = 1 / (len_doc_recip ** geometric_scaling)

        poly_pow = 0
        for i in range(hess_size):
            poly_pow = poly_pow + log(weights[start + i]) * counts[start + i]

        for i in range(hess_size):
            w_i = indices[start + i]
            counts_i = counts[start + i]
            totals_i = totals[start + i]
            weights_i = weights[start + i]

            jac_i = (poly_pow +
                     log(counts_i) -
                     log(weights_i))

            # Geometric scaling is applied to both the jacobian and hessian.
            # Multiplying because we're still in log space.
            # jac_i_norm = jac_i * len_doc_recip
            jac_i_norm = jac_i * geometric_scaling

            jacobian[w_i] = (jacobian[w_i] +
                             exp(jac_i_norm))

            for j in range(hess_size):
                hess_i_j = exp(jac_i +
                               log(counts[start + j]) -
                               log(weights[start + j]))

                if i == j:
                    hess_i_j *= (counts_i - 1) / counts_i

                # Geometric scaling; out of log space so it's a power now.
                # Note that when I replaced this with a simple square root,
                # everything worked about as well as the best
                # geometric_scaling parameter, and with all other settings
                # exactly the same. There is a chance this is just a very
                # elaborate way of taking the square root... but I don't
                # understand why we would need to take the square root,
                # whereas the geometric mean seems somewhat reasonable.
                # hess_i_j = hess_i_j ** len_doc_recip
                hess_i_j = hess_i_j ** geometric_scaling

                if arithmetic_norm > 0:
                    hess_i_j = hess_i_j / totals_i
                if isnan(hess_i_j) or isinf(hess_i_j):
                    continue

                w_j = indices[start + j]
                for k in range(embed_size):
                    embed_dot = (rand_hess[w_i, k] +
                                 hess_i_j *
                                 hash_vecs[w_j, k])
                    rand_hess[w_i, k] = embed_dot

        start = end

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void cy_train_chunk_vanilla_full(
        double[:] fout,
        double[:] jacobian,
        double[:, :] rand_hess,
        long[:] ends,
        long[:] indices,
        double[:] counts,
        double[:] totals,
        double[:] weights,
        double[:, :] hash_vecs) nogil:

    # Memorably named temp variables.
    cdef double poly_pow = 0
    cdef double jac_i = 0
    cdef double jac_i_norm = 0
    cdef double counts_i = 0
    cdef double totals_i = 0
    cdef double weights_i = 0
    cdef double hess_i_j = 0
    cdef double embed_dot = 0

    # Boundaries and index variables.
    cdef long n_docs = ends.shape[0]
    cdef long start = 0
    cdef long end = 0
    cdef long hess_size = 0
    cdef long embed_size = hash_vecs.shape[1]
    cdef long i, j, k, w_i, w_j, doc_ix

    # This performs a less flexible version of the operation performed by
    # `cy_train_chunk_combined`, but removes anything without a handwavy
    # justification, and also returns the value of the function itself
    # in addition to the hessian and jacobian. Most detailed notes are
    # omitted entirely; for documentation, see the comments to the main
    # version of this function above.

    for doc_ix in range(n_docs):
        end = ends[doc_ix]
        hess_size = end - start

        # Calclulate the value of the polynomial term in log space.
        poly_pow = 0
        for i in range(hess_size):
            poly_pow = poly_pow + log(weights[start + i]) * counts[start + i]

        # Briefly drop out of log space to save the scalar output
        # of the polynomial term.
        fout[0] = fout[0] + exp(poly_pow)

        for i in range(hess_size):
            w_i = indices[start + i]
            counts_i = counts[start + i]
            totals_i = totals[start + i]
            weights_i = weights[start + i]

            # Calculate the value of the jacobian by dividing out the
            # value of the given variable (named `weights_i` here) and
            # multiplying the term by the exponent (named `counts_i` here).
            jac_i = (poly_pow +
                     log(counts_i) -
                     log(weights_i))

            # Briefly drop out of log space to update jacobian
            jacobian[w_i] = (jacobian[w_i] + exp(jac_i))

            for j in range(hess_size):
                # Caclulate a hessian term by repeating the process used to
                # calculate the jacobian.
                hess_i_j = exp(jac_i +
                               log(counts[start + j]) -
                               log(weights[start + j]))

                # If we are on a diagonal, we need a nonlinear correction
                # to get the second partial derivative instead of the mixed
                # partial derivative. (Second derivative of x^3 is 6x, but
                # without this correction we would get 9x.)
                if i == j:
                    hess_i_j *= (counts_i - 1) / counts_i

                # Drop unexpected NANs.
                if isnan(hess_i_j) or isinf(hess_i_j):
                    continue

                # Perform the random projection.
                w_j = indices[start + j]
                for k in range(embed_size):
                    embed_dot = (rand_hess[w_i, k] +
                                 hess_i_j *
                                 hash_vecs[w_j, k])
                    rand_hess[w_i, k] = embed_dot

        start = end

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void cy_train_chunk_exponential_full(
        double[:] fout,
        double[:] jacobian,
        double[:, :] rand_hess,
        long[:] ends,
        long[:] indices,
        double[:] counts,
        double[:] totals,
        double[:] weights,
        double[:, :] hash_vecs) nogil:

    # Memorably named temp variables.
    cdef double poly_pow = 0
    cdef double jac_i = 0
    cdef double jac_i_norm = 0
    cdef double counts_i = 0
    cdef double totals_i = 0
    cdef double weights_i = 0
    cdef double hess_i_j = 0
    cdef double embed_dot = 0

    # Boundaries and index variables.
    cdef long n_docs = ends.shape[0]
    cdef long start = 0
    cdef long end = 0
    cdef long hess_size = 0
    cdef long embed_size = hash_vecs.shape[1]
    cdef long i, j, k, w_i, w_j, doc_ix

    # This performs a less flexible version of the operation performed by
    # `cy_train_chunk_combined`, but removes anything without a handwavy
    # justification, and also returns the value of the function itself
    # in addition to the hessian and jacobian. Most detailed notes are
    # omitted entirely; for documentation, see the comments to the main
    # version of this function above.

    for doc_ix in range(n_docs):
        end = ends[doc_ix]
        hess_size = end - start

        # Calclulate the value of the polynomial term in log space.
        poly_pow = 0
        for i in range(hess_size):
            poly_pow = poly_pow + log(weights[start + i]) * counts[start + i]

        # Briefly drop out of log space to save the scalar output
        # of the polynomial term.
        fout[0] = fout[0] + exp(poly_pow)

        for i in range(hess_size):
            w_i = indices[start + i]
            counts_i = counts[start + i]
            totals_i = totals[start + i]
            weights_i = weights[start + i]

            # Calculate the value of the jacobian by dividing out the
            # value of the given variable (named `weights_i` here) and
            # multiplying the term by the exponent (named `counts_i` here).
            jac_i = (poly_pow +
                     log(counts_i) -
                     log(weights_i))

            # Briefly drop out of log space to update jacobian
            jacobian[w_i] = (jacobian[w_i] + exp(jac_i) * weights_i)

            for j in range(hess_size):
                # Caclulate a hessian term by repeating the process used to
                # calculate the jacobian.
                hess_i_j = exp(jac_i +
                               log(counts[start + j]) -
                               log(weights[start + j]))

                # If we are on a diagonal, we need a nonlinear correction
                # to get the second partial derivative instead of the mixed
                # partial derivative. (Second derivative of x^3 is 6x, but
                # without this correction we would get 9x.)
                if i == j:
                    hess_i_j *= (counts_i - 1) / counts_i

                # Drop unexpected NANs.
                if isnan(hess_i_j) or isinf(hess_i_j):
                    continue

                # The "exponential" part. This turns the ordinary hessian
                # into the top part of the expression for an "elasticity"
                # hessian. I didn't totally make this up: elasticity is
                # a concept in economics, equivalent to d log x / d log y.
                # In a single-variable setting, it simplifies to
                #       x * f'(x) / f(x)
                # Here it's a little more complicated, but the idea is the
                # same. For the jacobian we can get away with doing this
                # later, but since we're doing random projection now for
                # the hessian, we have to do it here.

                hess_i_j = hess_i_j * weights_i * weights[start + j]

                # Perform the random projection.
                w_j = indices[start + j]
                for k in range(embed_size):
                    embed_dot = (rand_hess[w_i, k] +
                                 hess_i_j *
                                 hash_vecs[w_j, k])
                    rand_hess[w_i, k] = embed_dot

        start = end



@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef void cy_train_chunk(
        long[:] ends,
        long[:] indices,
        double[:] counts,
        double[:] totals,
        double[:] weights,
        double[:, :] hess_buffer,
        double[:, :] rand_hess,
        signed char[:, :] hash_vecs) nogil:
    cdef long start = 0
    cdef long end = 0
    cdef long doc_ix = 0
    cdef long n_docs = ends.shape[0]

    for doc_ix in range(n_docs):
        end = ends[doc_ix]

        cy_embed_hess(counts[start: end],
                      weights[start: end],
                      totals[start: end],
                      indices[start: end],
                      hess_buffer,
                      rand_hess,
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
        double[:, :] rand_hess,
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
                rand_hess[w_i, k] = (rand_hess[w_i, k] +
                                     hess_norm * hash_vecs[w_j, k])

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
    ends, indices, counts, totals, weights = docarray.features()
    random_hessian_projection = numpy.zeros(hash_vectors.shape,
                                            dtype=numpy.float64)

    cy_train_chunk_combined(ends, indices, counts, totals,
                            weights, random_hessian_projection, hash_vectors)

    nonzero_ix = random_hessian_projection.sum(axis=1) > 0
    return random_hessian_projection[nonzero_ix], nonzero_ix.nonzero()

def train_chunk_configurable_scaling(docarray,
                                     hash_vectors,
                                     random_hessian_projection_out,
                                     random_hessian_projection_buf,
                                     jacobian_out,
                                     jacobian_buf,
                                     geometric_scaling,
                                     arithmetic_norm=False,
                                     cosine_norm=False):
    ends, indices, counts, totals, weights = docarray.features()
    n_docs = len(ends)
    random_hessian_projection = numpy.zeros(hash_vectors.shape,
                                            dtype=numpy.float64)
    jacobian = numpy.zeros(hash_vectors.shape[0], dtype=numpy.float64)

    arithmetic_norm = int(bool(arithmetic_norm))
    cy_train_chunk_configurable_scaling(ends, indices, counts,
                                        totals, weights,
                                        jacobian, random_hessian_projection,
                                        hash_vectors, geometric_scaling,
                                        arithmetic_norm)

    if cosine_norm:
        embed_norm = (random_hessian_projection *
                      random_hessian_projection).sum(axis=1)[:, None] ** 0.5
        embed_norm[embed_norm == 0] = 1
        random_hessian_projection /= embed_norm
    random_hessian_projection /= n_docs * 10000

    with random_hessian_projection_buf.get_lock():
        random_hessian_projection_out += random_hessian_projection

    with jacobian_buf.get_lock():
        jacobian_out += jacobian

def train_chunk_vanilla_full(docarray,
                             hash_vectors,
                             fout_out,
                             fout_buf,
                             jacobian_out,
                             jacobian_buf,
                             random_hessian_projection_out,
                             random_hessian_projection_buf):
    ends, indices, counts, totals, weights = docarray.features()
    n_docs = len(ends)

    fout = numpy.zeros((1,), dtype=numpy.float64)
    random_hessian_projection = numpy.zeros(hash_vectors.shape,
                                            dtype=numpy.float64)
    jacobian = numpy.zeros(hash_vectors.shape[0], dtype=numpy.float64)


    cy_train_chunk_vanilla_full(fout, jacobian, random_hessian_projection,
                                ends, indices, counts, totals, weights,
                                hash_vectors)

    with fout_buf.get_lock():
        fout_out += fout

    with jacobian_buf.get_lock():
        jacobian_out += jacobian

    with random_hessian_projection_buf.get_lock():
        random_hessian_projection_out += random_hessian_projection

def train_chunk_exponential_full(docarray,
                                 hash_vectors,
                                 fout_out,
                                 fout_buf,
                                 jacobian_out,
                                 jacobian_buf,
                                 random_hessian_projection_out,
                                 random_hessian_projection_buf):
    ends, indices, counts, totals, weights = docarray.features()
    n_docs = len(ends)

    fout = numpy.zeros((1,), dtype=numpy.float64)
    random_hessian_projection = numpy.zeros(hash_vectors.shape,
                                            dtype=numpy.float64)
    jacobian = numpy.zeros(hash_vectors.shape[0], dtype=numpy.float64)


    cy_train_chunk_exponential_full(fout, jacobian, random_hessian_projection,
                                    ends, indices, counts, totals, weights,
                                    hash_vectors)

    with fout_buf.get_lock():
        fout_out += fout

    with jacobian_buf.get_lock():
        jacobian_out += jacobian

    with random_hessian_projection_buf.get_lock():
        random_hessian_projection_out += random_hessian_projection


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
    random_hessian_projection = numpy.zeros(hash_vectors.shape, dtype=numpy.float64)
    hess_buffer = numpy.zeros((50, 50), dtype=numpy.float64)

    for indices, counts, totals, weights in docarray:
        len_doc = counts.sum()
        if len_doc < 2:
            continue

        if len_doc > hess_buffer.shape[0]:
            newshape = (int(len_doc * 2), int(len_doc * 2))
            hess_buffer = numpy.zeros(newshape, dtype=numpy.float64)

        cy_embed_hess(counts, weights, totals, indices,
                      hess_buffer, random_hessian_projection, hash_vectors)

    nonzero_ix = random_hessian_projection.sum(axis=1) > 0
    return random_hessian_projection[nonzero_ix], nonzero_ix.nonzero()

def block_logspace_hessian(pow_vector, x_vector):
    cdef long size = x_vector.shape[0]
    out_arr = numpy.zeros((size, size), dtype=numpy.float64)
    cy_block_logspace_hessian(pow_vector, x_vector, out_arr)
    return out_arr
