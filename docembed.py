import numpy
import multiprocessing
import ctypes

from collections import Counter
from collections import abc
from itertools import islice

import psutil
from nltk.corpus import wordnet

import util

def _buffered_array_copy(array):
    new_array, buff = _buffered_array(array.shape,
                                      array.dtype)
    new_array[:] = array
    return new_array, buff

def _buffered_array(shape, dtype):
    dtype = numpy.dtype(dtype)
    size = 1
    for s in shape:
        size *= s

    n_bytes = size * dtype.itemsize
    buff = multiprocessing.Array(ctypes.c_byte, n_bytes)
    new_array = numpy.frombuffer(buff.get_obj(), dtype=dtype)
    return new_array.reshape(shape), buff

class ExtendWrap(abc.Sequence):
    def __init__(self, array):
        if isinstance(array, ExtendWrap):
            array = array._array
        new_array, buff = _buffered_array_copy(array)
        self._buffer = buff
        self._array = new_array
        self._len = new_array.shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.array[index]

    def __setitem__(self, index, value):
        self.array[index] = value

    @property
    def array(self):
        return self._array[:self._len]

    def trim(self):
        # Remove any trailing empty space in the underlying array. This
        # can be done by doing a `_buffered_array_copy` of the `array`
        # property, which has the desired size. (So don't use `_array` here.)
        self._array, self._buffer = _buffered_array_copy(self.array)

    def extend(self, new):
        old_len = self._len
        new_len = old_len + len(new)
        if new_len > self._array.shape[0]:
            new_shape = (new_len * 2,) + self._array.shape[1:]

            new_array, buff = _buffered_array(new_shape, self._array.dtype)
            new_array[:old_len] = self.array

            self._array = new_array
            self._buffer = buff

        self._array[old_len:new_len] = new
        self._len = new_len

class DocArray(abc.Sequence):
    def __init__(self, documents=None, ambiguity_vector='', 
                 ambiguity_scale=1.0 / 3, ambiguity_base=1.0,
                 hash_dimension=300, max_vocab=500000,
                 downsample_threshold=1.0):
        self.ambiguity_vector = ambiguity_vector
        self.ambiguity_scale = ambiguity_scale
        self.ambiguity_base = ambiguity_base
        self.hash_dimension = hash_dimension
        self.max_vocab = max_vocab
        self.downsample_threshold = downsample_threshold
        self._hash_vectors = None

        if isinstance(documents, DocArray):
            # Should we inherit settings from the DocArray passed in here??
            self.words = list(documents.words)
            self.word_index = dict(documents.word_index)
            self._word_count = ExtendWrap(documents._word_count)
            self._word_count_total = documents._word_count.array.sum()
            self._word_weight = ExtendWrap(documents._word_weight)
            self._ends = ExtendWrap(documents._ends)
            self._indices = ExtendWrap(documents._indices)
            self._counts = ExtendWrap(documents._counts)
        else:
            self.empty_index()

            if documents is not None:
                if isinstance(documents, abc.Sequence):
                    self.extend(documents)
                else:
                    self.extend_iter(documents)

    def __len__(self):
        return len(self._ends)

    def __getitem__(self, index):
        if isinstance(index, slice):
            if index.stop > len(self):
                index = slice(index.start, len(self), index.step)
            ends = self._ends[index]
            starts = numpy.zeros(ends.shape, dtype=ends.dtype)
            starts[1:] = ends[:-1]
            if index.start > 0:
                starts[0] = self._ends[index.start - 1]

            new_index = numpy.array([ix for s, e in zip(starts, ends)
                                     for ix in range(s, e)])
            new_ends = (ends - starts).cumsum()

            new = DocArray()
            new.inherit_settings(self)

            new.words = self.words
            new.word_index = self.word_index
            new._ends.extend(new_ends)
            new._word_count.extend(self._word_count)
            new._word_count_total = self._word_count_total
            new._word_weight.extend(self._word_weight)
            new._indices.extend(self._indices[new_index])
            new._counts.extend(self._counts[new_index])
            return new
        else:
            start = 0 if index == 0 else self._ends[index - 1]
            end = self._ends[index]
            return (self._indices[start:end],
                    self._counts[start:end],
                    self._word_count[self._indices[start:end]],
                    self._word_weight[self._indices[start:end]])

    def __add__(self, right):
        left = DocArray(self)
        left.inherit_settings(self)
        left.extend(right)
        return left

    def inherit_settings(self, docarray):
        self.ambiguity_vector = docarray.ambiguity_vector
        self.hash_dimension = docarray.hash_dimension

    @property
    def hash_vectors(self):
        if self._hash_vectors is None:
            self._hash_vectors = util.srp_matrix(self.words[:self.max_vocab],
                                                 self.hash_dimension)
        return self._hash_vectors

    @property
    def mean_doc_length(self):
        return numpy.diff(self._ends.array).mean()

    # Reset the word index. This also empties out the documents, since they
    # are useless wihtout the word index.
    def empty_index(self):
        self.words = []
        self.word_index = {}

        self._word_count = ExtendWrap(
            numpy.zeros(0, dtype=numpy.float64))
        self._word_count_total = 0
        self._word_weight = ExtendWrap(
            numpy.zeros(0, dtype=numpy.float64))

        self.empty_docs()

    # Empty out the docarray, but preserve the word indices and counts
    # if they exist.
    def empty_docs(self):
        self._ends = ExtendWrap(
            numpy.zeros(0, dtype=numpy.int64))
        self._indices = ExtendWrap(
            numpy.zeros(0, dtype=numpy.int64))
        self._counts = ExtendWrap(
            numpy.zeros(0, dtype=numpy.float64))

    def features(self):
        return (self._ends.array,
                self._indices.array,
                self._counts.array,
                self._word_count.array[self._indices],
                self._word_weight.array[self._indices])

    def str_count(self, word_string):
        return self._word_count[self.word_index[word_string]]

    def iter_as_documents(self):
        for indices, counts, totals, weights in self:
            yield [w
                   for ix, ct in zip(indices, counts)
                   for w in [self.words[ix]] * ct]

    def add_words(self, words):
        new = list(set(words) - self.word_index.keys())
        ignore = []
        if len(new) + len(self.words) > self.max_vocab:
            last_n = self.max_vocab - len(self.words)
            last_n = last_n if last_n > 0 else 0
            ignore = new[last_n:]
            new = new[:last_n]

        old_len = len(self.words)
        if new:
            for i, w in enumerate(new):
                self.words.append(w)
                self.word_index[w] = i + old_len
            self._word_count.extend([0] * len(new))
            self._word_weight.extend([0] * len(new))

            self._hash_vectors = None

        if ignore:
            for i, w in enumerate(ignore):
                self.words.append(w)
                self.word_index[w] = self.max_vocab - 1

    def extend_iter(self, doc_iter, chunksize=10000):
        if isinstance(doc_iter, DocArray):
            doc_iter = doc_iter.iter_as_documents()

        chunks = [c for c in islice(doc_iter, chunksize) if c]
        while chunks:
            self.extend(chunks)
            chunks = [c for c in islice(doc_iter, chunksize) if c]

    def extend(self, documents):
        if isinstance(documents, DocArray):
            self.extend_iter(documents)
        else:
            documents = [doc for doc in documents if doc]
            self.add_words(w for doc in documents for w in doc)

            wix = self.word_index
            document_counters = [Counter([wix[w] for w in doc])
                                 for doc in documents]
            document_counts = [c.items() for c in document_counters]

            total = Counter(wix[w] for doc in documents for w in doc)
            batch_total = sum(total.values())
            self._word_count_total += batch_total
            word_count_total = self._word_count_total
            mean_word_freq = word_count_total / len(self._word_count)
            sq_word_prob_sum = sum((c / word_count_total) ** 2
                                   for c in self._word_count)

            for ix, ct in total.items():
                self._word_count[ix] += ct

            # Only update word weights for new words.
            to_update = [ix for ix, ct in total.items() 
                         if self._word_weight[ix] <= 0]

            # TODO: vectorize this where possible
            if self.ambiguity_vector in ('', '1', 'one'):
                for ix in to_update:
                    self._word_weight[ix] = 1
            elif self.ambiguity_vector == 'scalefree':
                for ix in to_update:
                    self._word_weight[ix] = \
                        numpy.exp(self._word_count[ix] / word_count_total -
                                  sq_word_prob_sum)
            else:
                if self.ambiguity_vector == 'log':
                    values = [numpy.log10(self._word_count[ix]) * self.ambiguity_scale + 1
                              for ix in to_update]
                elif self.ambiguity_vector == 'wordnet':
                    synset_lens = (len(wordnet.synsets(self.words[ix]))
                                   for ix in to_update)
                    values = [numpy.log10(syn_len + 1) *
                              self.ambiguity_scale + 1
                              for syn_len in synset_lens]
                
                self._word_weight.array[to_update] = values

            indices, counts = zip(*[i_c
                                    for doc in document_counts
                                    for i_c in doc])
            counts = numpy.array(counts, dtype=float)
            ends = numpy.cumsum([len(doc) for doc in document_counts])

            if self.downsample_threshold < 1.0:
                threshold = self.downsample_threshold
                down_by_p = {
                    word_i: 1 - (batch_total * threshold / total[word_i]) ** 0.5
                    for word_i, w in enumerate(self.words[:self.max_vocab])
                    if (total[word_i] / batch_total) > threshold
                }
                print('  Number of word types to be downsampled: ', len(down_by_p))

                to_downsample = ((doc_i,) * int(counts[doc_i]) for doc_i in range(len(indices)) 
                                 if indices[doc_i] in down_by_p)

                to_downsample = numpy.array([doc_i for doc_i_set in to_downsample for doc_i in doc_i_set])
                rand = numpy.random.random(len(to_downsample))
                sample_thresh = numpy.array([down_by_p[indices[doc_i]] for doc_i in to_downsample])
                to_downsample = to_downsample[rand < sample_thresh]
                for doc_i in to_downsample:
                    counts[doc_i] -= 1
                counts[counts <= 0] = 1e-10

            self._ends.extend(ends + len(self._indices))
            self._indices.extend(indices)
            self._counts.extend(counts)

            ww_rav = self._word_weight.array.ravel()
            ww_max = ww_rav.max()
            ww_min = ww_rav.min()
            ww_avg = ww_rav.mean()
            print('  Word weights after new words added:')
            print(f'    weight max: {ww_max:.4g}  min: {ww_min:.4g}  avg: {ww_avg:.4g}')
            print()

    def overwrite(self, documents):
        self.empty_docs()
        self.extend(documents)

# OK: This word embedding model starts from the assumption that
# words are types, and that individual instances of words --
# a.k.a. tokens -- are distinct values of the corresponding type.
# Sentences are composed of these word types in an order-
# agnostic way, meaning that type multiplication is
# commutative and distributes over type addition in the
# ordinary way. To calculate the model, we take the
# *type-level* second derivative for each pair of word types.
# The result is a hessian matrix of two-hole context types.

# This hessian matrix gives us a natural vector representation
# for each word type. In this vector space, each dimension
# corresponds to a word type, and the magnitude of the vector
# in that dimension corresponds to the value of the hessian
# when reinterpreted as the hessian of a polynomial that
# that calculates the *magnitude* of each sentence type and
# sums them together. This polynomial is, in effect, the
# polynomial describing the total number of possible sentences
# in the language given that each type has some (variable)
# number of values (tokens). A given position in this space
# corresponds to a fixed number of tokens for each word type
# dimension, and the hessian at that point goves us the
# rate of sentence type creation or destruction when we vary
# the position by an "infinitesimal" amount along the two
# given word axes. Infinitesimal is in quotes because these
# are discrete types, and it might at first seem more
# appropriate to take the discrete difference. The fact that
# we don't have to is related to ideas from synthetic
# differetnial geometry and smooth infinitesimal analysis.
# [Insert a bunch of category theory I don't know here.]
# In effect, this is the best linear approximation of the
# discrete difference.

# To make this really, really concrete, consider a language
# with just three word types, /a/, /the/, and /of/, and just
# two sentences: /a the of/, and /a the the/. Our type-level
# polynomial is

# /a/ ^ 1 * /the/ ^ 1 * /of/ ^ 1 +
# /a/ ^ 1 * /the/ ^ 2

# Given a function `count(/t/)` that takes a finite type
# and counts the number of possible values for the type, the
# corresponding magnitude polynomial can be constructed like
# so:

# count(/a/) * count(/the/) * count(/of/) +
# count(/a/) * count(/the/) ^ 2

# The hessian of this polynomial is a symmetric 3x3 matrix:

#     /a/                      /the/                             /of/
# /a/  0   c(/a/) * c(/of) + 2 * c(/a/) * c(/the/)    count(/a/) * count(/the/)
# ... and so on.

# This is easy to calculate because -- yay! -- partial
# differentiation (PD) is a linear operator, and so the PD
# of a sum of polynomials is just the sum of PDs of each
# individual polynomial. So we just take the respective
# PDs for each individual sentence type and add them up.

# At this point, we could start to talk about what the
# `count(/t/)` function should actually be counting. A
# reasonable approach might be to simply take the word
# count over a training set as the value. But before we
# bother with that, lets see what happens when we instead
# set `count(/t/)` to 1 for all types. This is obviously
# an oversimplification, but it turns out that it produces
# a calculation that is tractable and interesting. The
# value of the hessian at this point in word type space
# winds up being a simple co-occurrence count sum for
# each pair of words! This is obviously just a hop,
# skip, and jump away from the classic PPMI / shifted
# PMI matrix approach to generating word embeddings.
# But that takes us into undefined waters; remember that
# we didn't start out thinking probabilistically at all,
# and so we don't want to bring PMI into this. Also,
# we don't have to worry about negative values because
# we're just looking at counts; everything is positive
# already. There's also reason to suspect that because
# this is formulated on the basis of type calculus,
# the problems of thinking probabilistically about
# high-frequency words are being handled using other
# means. It may be that evaluating the derivative
# at a realistic point in the `count(/t/)` space
# would handle those problems even more effectively.
# I'm not sure; that's just a conjecture. So...

# Let's worry about more complex models later and see
# what we can do with this new way of looking at creating
# vector spaces with words. Since we have a pretty
# good justification for thinking about these embeddings
# as *vectors in a linear space* (and not just as vectors
# that magically estimate some probabilities), let's
# see what happens when we do dimension reduction, but
# instead of doing it using expensive things like
# word2vec, LSA, etc, we just use random projections
# for each word... so we project each word's vector
# into a reduced vector space, such that each word
# dimension corresponds to a -- literally! -- random
# set of smaller dimensions. One might think this could
# never work; surely we need to do SVD or something
# like that. But given that we now have a really strong
# justificaiton for the intiial vector space
# representation, we should feel OK giving it a shot.

# This operation turns bout to be mind-bogglingly
# simple to implement as a simple iterative counting
# algorithm, given some pre-calculated random word
# projections. And it works!

# If we were to simply iterate the vectors
# this way, normalizing dimensions by length, we'd
# get straight-up power iteration, and, hence, the
# first step of SVD. We want to avoid having all the
# vectors collapse towards the single largest
# eigenvector, though. One way to avoid that is to
# take the newly calculated embedding vectors and
# sparsify them so that they "look like" the original
# binary projection vectors.

# But in the end, the most effective approach might
# be to be to perform SVD over a small set of random
# projection vectors. I was doing this and it was
# working! But then it stopped working... and I
# cannot for the life of me figure out what I
# changed.

# For multiprocessing, the following functions must
# be in the global namespace of this module.

def train_chunk_multi(slice):
    err_count = util.train_chunk_vanilla_full(
        MP_DOC_ARRAY[slice], MP_HASH_VECS,
        MP_FOUT_OUT, MP_FOUT_BUF,
        MP_JACOB_OUT, MP_JACOB_BUF,
        MP_EMBED_OUT, MP_EMBED_BUF)
    if err_count > 0:
        print()
        print("Number of NAN values returned:", err_count)

class Embedding(object):
    def __init__(self, docarray=None, verbose=False):
        self.docarray = None
        self.verbose = verbose
        self._erase_on_reset = False

        if docarray is None:
            self.append_docarray(DocArray())
        else:
            self.append_docarray(docarray)

    @property
    def n_bits(self):
        return self.docarray.hash_dimension

    @property
    def hash_vectors(self):
        return self.docarray.hash_vectors

    def append_docarray(self, docarray):
        if self.docarray is not None:
            self.docarray.extend(docarray)
        else:
            self.docarray = docarray

        self.hash_iter_vectors = self.hash_vectors.copy()
        self._new_embedding()

    def overwrite_docarray(self, docarray):
        if not self.docarray:
            self.append_docarray(docarray)
        else:
            self.docarray.overwrite(docarray)
            self._extend_embedding()
            self.hash_iter_vectors = self.hash_vectors.copy()

    def _new_embedding(self):
        fout, buf = _buffered_array((1,),
                                    dtype=numpy.float64)
        fout[:] = 0
        self.fout = fout
        self.fout_buffer = buf

        jac, jac_buf = _buffered_array((self.hash_vectors.shape[0],),
                                       dtype=numpy.float64)
        jac[:] = 0
        self.jacobian_vector = jac
        self.jacobian_vector_buffer = jac_buf

        emb, buf = _buffered_array(self.hash_vectors.shape,
                                   dtype=numpy.float64)
        emb[:] = 0
        self.embed_vectors = emb
        self.embed_vectors_buffer = buf

        emb, buf = _buffered_array(self.hash_vectors.shape,
                                   dtype=numpy.float64)
        emb[:] = 0
        self.embed_vectors_accumulated = emb
        self.embed_vectors_accumulated_buffer = buf

        self.n_batches_complete = 0

    def _extend_embedding(self):
        old_emb_mean = self.embed_vectors_accumulated
        old_emb = self.embed_vectors
        old_jac = self.jacobian_vector
        old_fout = self.fout[0]
        old_n_batches = self.n_batches_complete

        self._new_embedding()
        self.embed_vectors_accumulated[:len(old_emb_mean)] = old_emb_mean
        self.embed_vectors[:len(old_emb)] = old_emb
        self.jacobian_vector[:len(old_jac)] = old_jac
        self.fout[0] = old_fout
        self.n_batches_complete = old_n_batches

    def _reset_embedding(self):
        if self._erase_on_reset:
            self._new_embedding()

    def get_vec(self, word):
        return self.embed_vectors[self.docarray.word_index[word]]

    def get_vecs(self, words):
        return [self.embed_vectors[self.docarray.word_index[w]]
                for w in words]

    def get_dist_func(self, word, euclidean=False):
        word_vec = word
        if isinstance(word, str):
            word_vec = self.get_vec(word)

        if euclidean:
            def dist(w):
                return util.euclidean_dist(self.get_vec(w), word_vec)
        else:
            def dist(w):
                return 1 - util.cosine_sim(self.get_vec(w), word_vec)

        return dist

    def closest_words(self, word, n_words, euclidean=False, mincount=1):
        return sorted(
            [w for w in self.docarray.words
             if self.docarray.str_count(w) >= mincount],
            key=self.get_dist_func(word, euclidean)
        )[:n_words]

    def analogy(self, positive, negative, n_words, euclidean=False, mincount=1):
        vec = numpy.add.reduce([self.get_vec(w) for w in positive])
        vec -= numpy.add.reduce([self.get_vec(w) for w in negative])
        return self.closest_words(vec, n_words, euclidean, mincount)

    def interpret_dimension(self, dim, n_words, euclidean=False, mincount=1):
        vec = numpy.zeros(self.embed_vectors.shape[1])
        vec[dim] = 1
        return self.closest_words(vec, n_words, euclidean, mincount)

    def _svd_prep(self, U, s):
        return U[:, :len(s)], numpy.diag(s)

    def _svd_test(self, test_matrix):
        U, s, V = numpy.linalg.svd(test_matrix)
        U_, s_ = self._svd_prep(U, s)

        # Sanity checks...

        # Does U @ s @ V reconstruct the data?
        assert numpy.allclose(U_ @ s_ @ V, test_matrix, atol=1e-05)

        # Does X @ V.T == U @ s?
        assert numpy.allclose(U_ @ s_, test_matrix @ V.T)

        return U, s, V

    def _svd_sq_test(self, raw_matrix, full_test=False):
        if full_test:
            mean = raw_matrix.mean(axis=0)
            std = raw_matrix.std(axis=0)

            matrix = raw_matrix - mean
            matrix /= std
        else:
            matrix = raw_matrix

        U_sq, s_sq, V_sq = self._svd_test(matrix.T @ matrix)
        result = matrix @ V_sq.T

        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        neg = matrix < 0
        print(matrix[neg].ravel().sum())
        print((mean * mean).sum() ** 0.5)
        print((std * std).sum() ** 0.5)

        if full_test:
            U, s, V = self._svd_test(matrix)
            assert numpy.allclose(s * s, s_sq, atol=1e-05)

            U, s = self._svd_prep(U, s)
            assert numpy.allclose(numpy.abs(U @ s),
                                  numpy.abs(result))

        return result, s_sq

    def _eigen_embedding(self):
        ev = self.embed_vectors
        ev = ev * ((ev * ev).sum(axis=1) ** 0.5)  # Cosine distance norm
        sample = numpy.random.choice(len(ev), size=ev.shape[1] * 20)
        ev_sample = (ev[sample])

        # Thorough test on a subsample
        result, s = self._svd_sq_test(ev_sample)

        # If the above test passes, then we can pass the full set of
        # embedding vectors, and skip the full test, which will run
        # out of memory on the full set.
        result, s = self._svd_sq_test(ev, full_test=False)

        print('  -- SVD --')
        print('Shape of embedding vectors:')
        print('{} rows, {} columns'.format(*ev.shape))
        print('First 10 values of s:')
        print(s[:10])
        print('Last 10 values of s:')
        print(s[-10:])

        return result, s

    def _chunkparams(self, embed_shape):
        # Eventually, try to guess how much memory is available and try
        # to balance chunksize and n_procs against avialable memory.
        
        mem_gigs_avail = psutil.virtual_memory().available
        n_words, n_dims = embed_shape
        min_chunksize = 1000
        mean_doc_length = int(self.docarray.mean_doc_length) + 1
        max_chunksize = int((mem_gigs_avail * 2 ** 23 // mean_doc_length) // n_dims)
        n_docs = len(self.docarray)
        n_procs = multiprocessing.cpu_count() * 2
        chunksize = n_docs // n_procs + 1
        chunksize = max(min_chunksize, chunksize)
        chunksize = min(max_chunksize, chunksize)
        if chunksize == min_chunksize:
            n_procs = (n_docs - 1) // min_chunksize + 1
        return chunksize, n_procs

    def _chunkslicer(self, all_chunks, n_procs):
        def chunkslicer():
            return tuple(islice(all_chunks, 0, n_procs))
        return iter(chunkslicer, ())

    def train_multi(self,
                    cosine_norm=False):
        self._reset_embedding()

        chunksize, n_procs = self._chunkparams(self.hash_iter_vectors.shape)

        global MP_DOC_ARRAY, MP_HASH_VECS
        global MP_FOUT_OUT, MP_FOUT_BUF
        global MP_JACOB_OUT, MP_JACOB_BUF
        global MP_EMBED_OUT, MP_EMBED_BUF
        global MP_GEOM_SCALE, MP_ARITH_NORM, MP_COSINE_NORM

        MP_DOC_ARRAY = self.docarray
        MP_HASH_VECS = self.hash_iter_vectors
        MP_FOUT_OUT = self.fout
        MP_FOUT_BUF = self.fout_buffer
        MP_JACOB_OUT = self.jacobian_vector
        MP_JACOB_BUF = self.jacobian_vector_buffer
        MP_EMBED_OUT = self.embed_vectors
        MP_EMBED_BUF = self.embed_vectors_buffer
        MP_COSINE_NORM = cosine_norm

        slices = (slice(i, i + chunksize)
                  for i in range(0, len(self.docarray), chunksize))

        n_docs = len(self.docarray)
        n_chunks = n_docs // chunksize + 1
        tenth = max(n_chunks // 10, 1)
        print("  Starting multiprocessing:")
        print("    {} processes".format(n_procs))
        print("    {} tasks per chunk".format(chunksize))
        print("    ~{} chunks".format(n_chunks))
        print("    completed: 0... ", end='', flush=True)

        with multiprocessing.Pool(processes=n_procs,
                                  maxtasksperchild=3) as pool:
            chunk_ct = 0
            for result in pool.imap_unordered(train_chunk_multi, slices):
                chunk_ct += 1
                if chunk_ct % tenth == 0:
                    if chunk_ct % (tenth * 5) == 0:
                        print()
                        print('               ', end='')
                    print('{}... '.format(chunk_ct),
                          end='',
                          flush=True)
            print()

        # The hessian of the log of a function is equal to the
        # hessian of the function divided by the value of the function,
        # minus the jacobian outer product divided by the squared value
        # of the function. This performs those additional operations
        # and reports a few simple statistics to verify that everything
        # is working as expected.

        fout = self.fout[0]

        if self.verbose:
            ev_rav = self.embed_vectors.ravel()
            ev_max = ev_rav.max()
            ev_min = ev_rav.min()
            ev_avg = ev_rav.mean()
            print( '    Before scaling by fout:')
            print(f'      fout: {fout}')
            print(f'      embed max: {ev_max:.4g}')
            print(f'      embed min: {ev_min:.4g}')
            print(f'      embed avg: {ev_avg:.4g}')

        embed_vectors = self.embed_vectors / fout
        
        if self.verbose:
            ev_rav = embed_vectors.ravel()
            ev_max = ev_rav.max()
            ev_min = ev_rav.min()
            ev_avg = ev_rav.mean()
            print( '    Before jacobian shift:')
            print(f'      embed max: {ev_max:.4g}')
            print(f'      embed min: {ev_min:.4g}')
            print(f'      embed avg: {ev_avg:.4g}')

        jacobian_norm = self.jacobian_vector / fout
        jacobian_rand = jacobian_norm @ self.hash_iter_vectors
        jacobian_rand = (jacobian_norm.reshape(-1, 1) @
                         jacobian_rand.reshape(1, -1))

        if self.verbose:
            jac_rav = jacobian_rand.ravel()
            jac_max = jac_rav.max()
            jac_min = jac_rav.min()
            jac_avg = jac_rav.mean()
            print( '    Jacobian shift matrix:')
            print(f'      shift max: {ev_max:.4g}')
            print(f'      shift min: {ev_min:.4g}')
            print(f'      shift avg: {ev_avg:.4g}')

        embed_vectors -= jacobian_rand

        if self.verbose:
            ev_rav = embed_vectors.ravel()
            ev_max = ev_rav.max()
            ev_min = ev_rav.min()
            ev_avg = ev_rav.mean()
            print( '    After jacobian shift:')
            print(f'      embed max: {ev_max:.4g}')
            print(f'      embed min: {ev_min:.4g}')
            print(f'      embed avg: {ev_avg:.4g}')


        self.n_batches_complete += 1
        if cosine_norm:
            norm = (embed_vectors ** 2).sum(axis=1) ** 0.5
            norm[norm == 0] = 1
            with self.embed_vectors_accumulated_buffer.get_lock():
                self.embed_vectors_accumulated += (embed_vectors / norm[:,None] -
                                                   self.embed_vectors_accumulated) / self.n_batches_complete
            with self.embed_vectors_buffer.get_lock():
                self.embed_vectors[:] = 0
            with self.jacobian_vector_buffer.get_lock():
                self.jacobian_vector[:] = 0
            with self.fout_buffer.get_lock():
                self.fout[:] = 0
            if self.verbose:
                ev_rav = self.embed_vectors_accumulated.ravel()
                ev_max = ev_rav.max()
                ev_min = ev_rav.min()
                ev_avg = ev_rav.mean()
                print( '    Accumulated unit-normalized mean:')
                print(f'      embed max: {ev_max:.4g}')
                print(f'      embed min: {ev_min:.4g}')
                print(f'      embed avg: {ev_avg:.4g}')
        else:
            with self.embed_vectors_accumulated_buffer.get_lock():
                self.embed_vectors_accumulated[:] = embed_vectors
        
        if not self.verbose:
            print(f'    Fout for this batch: {fout}')
            print( '    Final embedding stats:')
            ev_rav = self.embed_vectors_accumulated.ravel()
            ev_max = ev_rav.max()
            ev_min = ev_rav.min()
            ev_avg = ev_rav.mean()
            print(f'      embed max: {ev_max:.4g}  min: {ev_min:.4g}  avg: {ev_avg:.4g}')

        print()

    def _save_wordlist(self, mincount, out_vocab):
        tot = self.docarray._word_count
        wix = self.docarray.word_index
        words = sorted(self.docarray.words,
                       key=lambda w: tot[wix[w]],
                       reverse=True)
        words = [w for w in words
                 if tot[wix[w]] >= mincount and
                 wix[w] != (self.docarray.max_vocab - 1)]
        return words[:out_vocab]

    def save_vectors(self, filename, mincount=10, n_words=400_000):
        wix = self.docarray.word_index
        words = self._save_wordlist(mincount, n_words)

        embed_vectors = self.embed_vectors_accumulated.astype(numpy.float16)
        vecs = (embed_vectors[wix[w]] for w in words)

        sample_vec = embed_vectors[wix[words[0]]]

        rows = ('{} {}\n'.format(w, ' '.join(str(v) for v in vec))
                for w, vec in zip(words, vecs))

        with open(filename, 'w', encoding='utf-8') as op:
            for r in rows:
                op.write(r)

        return dict(zip(words, vecs))

    def save_vocab(self, filename, mincount=10, n_words=400_000):
        tot = self.docarray._word_count
        wix = self.docarray.word_index
        words = self._save_wordlist(mincount, n_words)

        rows = ('{} {}\n'.format(w, tot[wix[w]]) for w in words)
        with open(filename, 'w', encoding='utf-8') as op:
            for r in rows:
                op.write(r)


