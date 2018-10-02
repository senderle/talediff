import numpy
import multiprocessing
import psutil
from collections import Counter
from collections import abc
from itertools import islice
from sparsehess import block_logspace_hessian as block_logspace_hessian_sp
from sparsehess import train_chunk_py
from sparsehess import train_chunk_cy
from sparsehess import train_chunk_jacobian_cy
from sparsehess import train_chunk_configurable_cy

class ExtendWrap(abc.Sequence):
    def __init__(self, array):
        if isinstance(array, ExtendWrap):
            array = array._array.copy()
        self._array = array
        self._len = array.shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self.array[index]

    def __setitem__(self, index, value):
        self.array[index] = value

    @property
    def array(self):
        return self._array[:self._len]

    def trimmed(self):
        return self._array[:self._len].copy()

    def trim(self):
        self._array = self.trimmed()

    def extend(self, new):
        old_len = self._len
        new_len = old_len + len(new)
        if new_len > self._array.shape[0]:
            new_shape = (new_len * 2,) + self._array.shape[1:]
            array = numpy.zeros(new_shape, dtype=self._array.dtype)
            array[:old_len] = self.array
            self._array = array
        self._array[old_len:new_len] = new
        self._len = new_len

class DocArray(abc.Sequence):
    def __init__(self, documents=None, eval_mode=''):
        self.eval_mode = eval_mode

        if isinstance(documents, DocArray):
            self.words = list(documents.words)
            self.word_index = dict(documents.word_index)
            self._word_count = ExtendWrap(documents._word_count)
            self._log_word_count = ExtendWrap(documents._log_counts)
            self._ends = ExtendWrap(documents._ends)
            self._indices = ExtendWrap(documents._indices)
            self._counts = ExtendWrap(documents._counts)
        else:
            self.words = []
            self.word_index = {}
            self._word_count = ExtendWrap(
                numpy.zeros(0, dtype=numpy.float64))
            self._log_word_count = ExtendWrap(
                numpy.zeros(0, dtype=numpy.float64))
            self._ends = ExtendWrap(
                numpy.zeros(0, dtype=numpy.int64))
            self._indices = ExtendWrap(
                numpy.zeros(0, dtype=numpy.int64))
            self._counts = ExtendWrap(
                numpy.zeros(0, dtype=numpy.float64))

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

            new.words = self.words
            new.word_index = self.word_index
            new._ends.extend(new_ends)
            new._word_count.extend(self._word_count)
            new._log_word_count.extend(self._log_word_count)
            new._indices.extend(self._indices[new_index])
            new._counts.extend(self._counts[new_index])
            return new
        else:
            start = 0 if index == 0 else self._ends[index - 1]
            end = self._ends[index]
            return (self._indices[start:end],
                    self._counts[start:end],
                    self._word_count[self._indices[start:end]],
                    self._log_word_count[self._indices[start:end]])

    def __add__(self, right):
        left = DocArray(self)
        left.extend(right)
        return left

    def save_memmap(self):
        pass

    def load_memmap(self):
        pass

    def features(self):
        return (self._ends.array,
                self._indices.array,
                self._counts.array,
                self._word_count.array[self._indices],
                self._log_word_count.array[self._indices])

    def str_count(self, word_string):
        return self._word_count[self.word_index[word_string]]

    def iter_as_documents(self):
        for indices, counts, totals, log_totals in self:
            yield [w
                   for ix, ct in zip(indices, counts)
                   for w in [self.words[ix]] * ct]

    def add_words(self, words):
        new = set(words) - self.word_index.keys()
        if new:
            new = list(new)
            old_len = len(self.words)
            for i, w in enumerate(new):
                self.words.append(w)
                self.word_index[w] = i + old_len
            self._word_count.extend([0] * len(new))
            self._log_word_count.extend([0] * len(new))

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
            all_total = sum(total.values())
            mean_word_freq = all_total / len(total)
            for ix, ct in total.items():
                self._word_count[ix] += ct
                if self.eval_mode == 'log':
                    self._log_word_count[ix] = \
                        numpy.log10(self._word_count[ix]) + 1
                elif self.eval_mode == '1log':
                    self._log_word_count[ix] = \
                        1 / (numpy.log10(self._word_count[ix]) + 1)
                elif self.eval_mode == 'scalefree':
                    self._log_word_count[ix] = \
                        numpy.exp(self._word_count[ix] / all_total -
                                  mean_word_freq / all_total)
                elif self.eval_mode == '1scalefree':
                    self._log_word_count[ix] = \
                        numpy.exp(mean_word_freq / all_total -
                                  self._word_count[ix] / all_total)
                elif self.eval_mode in ('', '1', 'one'):
                    self._log_word_count[ix] = 1

            indices, counts = zip(*[i_c
                                    for doc in document_counts
                                    for i_c in doc])
            ends = numpy.cumsum([len(doc) for doc in document_counts])

            self._ends.extend(ends + len(self._indices))
            self._indices.extend(indices)
            self._counts.extend(counts)

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

class Embedding(object):
    def __init__(self, docarray=None, multiplier=10, sparsifier=-1):
        self.docarray = None
        self.multiplier = multiplier
        self.sparsifier = sparsifier
        self._erase_on_reset = False

        if docarray is None:
            self.append_docarray(DocArray())
        else:
            self.append_docarray(docarray)

    @property
    def n_bits(self):
        sp = max(self.sparsifier, 0)
        return 64 * self.multiplier * 2 ** sp

    def append_docarray(self, docarray):
        if self.docarray:
            self.docarray.extend(docarray)
        else:
            self.docarray = docarray
        self.hash_vectors = srp_matrix(self.docarray.words,
                                       self.multiplier,
                                       self.sparsifier)
        self.hash_iter_vectors = self.hash_vectors.copy()
        self.embed_vectors = numpy.zeros(self.hash_vectors.shape)

    def step_embedding(self):
        self.hash_iter_vectors = self._sparsify_embedding()
        self._erase_on_reset = True

    def _reset_embedding(self):
        if self._erase_on_reset:
            self.embed_vectors = numpy.zeros(self.hash_iter_vectors.shape)

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
                return euclidean_dist(self.get_vec(w), word_vec)
        else:
            def dist(w):
                return 1 - cosine_sim(self.get_vec(w), word_vec)

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

    def _sparsify_embedding(self):
        # The point of this is to recreate a binary project akin to
        # the one generated by srp_matrix, but taking account of whatever
        # new semantic information has been accrued by the embedding.
        # Without this step, the vectors start to overfit badly. (Or
        # at least that's what it looks like to me; really rare words
        # start popping up as good matches.)

        # We also have to be careful here because very common words
        # will wind up appearing in every vector, and will lose their
        # distinct character. As a result important distinguishing
        # informaiton about other words is lost. (I.e. a word is more
        # likely to be a noun if it appears along with "the" more
        # often, but if we can't tell between "the" and "if," that
        # information is lost.) So we need to have vectors for very
        # common words be more sparse than vectors for rare words.

        # It's not clear what approach solves these problems best.
        # The approach implemented in `resample_vectors_median`
        # worked OK, not great. The probabilistic approach implemented
        # in `resample_vectors` was better, but much slower.

        return resample_vectors(self.embed_vectors)

    def _eigen_embedding(self):
        ev = self.embed_vectors
        ev -= ev.mean(axis=0)
        U, s, V = numpy.linalg.svd(ev.T @ ev)
        result = ev @ U.T
        print('  -- SVD --')
        print('Shape of embedding vectors:')
        print('{} rows, {} columns'.format(*ev.shape))
        print('First 10 values of s:')
        print(s[:10])
        print('Last 10 values of s:')
        print(s[-10:])
        print('Shape of output vectors:')
        print('{} rows, {} columns'.format(*result.shape))
        return result

    def train(self):
        self._reset_embedding()
        embed = train_chunk(self.docarray, self.hash_iter_vectors)
        self.embed_vectors += embed

    def train_sp(self):
        self._reset_embedding()
        embed = train_chunk_cy(self.docarray, self.hash_iter_vectors)
        self.embed_vectors += embed

    def _chunkparams(self, embed_shape):
        # Eventually, try to guess how much memory is available and try
        # to balance chunksize and n_procs against avialable memory.
        # psutil.virtual_memory().available

        n_words, n_dims = embed_shape
        min_chunksize = 1000
        max_chunksize = 2 ** 25 // n_dims
        n_docs = len(self.docarray)
        n_procs = multiprocessing.cpu_count()
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
                    with_jacobian=False,
                    arithmetic_norm=False,
                    geometric_norm=False):
        self._reset_embedding()
        hash_vectors = self.hash_iter_vectors
        n_docs = len(self.docarray)
        chunksize, n_procs = self._chunkparams(hash_vectors.shape)

        # For reasons I do not fully understand, multiprocessing doesn't
        # schedule large chunks very well; we wind up using just two
        # or three processors at a time even when more are available.
        # So we divide our chunks into sub-chunks, and process each
        # sub-chunk in a new pool. This adds more overhead than I like
        # but it speeds up large processes according to my tests. There's
        # probably a more correct solution to this problem, but I don't
        # know what it is.

        all_chunks = ((self.docarray[i:i + chunksize], hash_vectors)
                      for i in range(0, len(self.docarray), chunksize))
        chunk_ct = 0
        print("Starting multiprocessing:")
        print("    {} processes".format(n_procs))
        print("    {} tasks per chunk".format(chunksize))
        print("    ~{} chunks".format(n_docs // chunksize + 1))

        jac = numpy.zeros(self.embed_vectors.shape[0], dtype=numpy.float64)
        if arithmetic_norm and geometric_norm:
            train = train_chunk_star_jac_arith_geom
        elif geometric_norm:
            train = train_chunk_star_jac_geom
        elif arithmetic_norm:
            train = train_chunk_star_jac_arith
        else:
            train = train_chunk_star_jac_unnormed

        for chunks in self._chunkslicer(all_chunks, n_procs):
            print('processing chunks {}-{}'.format(chunk_ct,
                                                   chunk_ct + len(chunks)))
            chunk_ct += len(chunks)
            with multiprocessing.Pool(processes=n_procs) as pool:
                for result in pool.imap(train, chunks):
                    ev_r, jac_r, ix_r = result
                    self.embed_vectors[ix_r] += ev_r
                    jac[ix_r] += jac_r

        if with_jacobian:
            jac[jac == 0] = 1
            self.embed_vectors /= jac[:, None]

    def train_simple(self):
        self._reset_embedding()
        hash_vectors = self.hash_iter_vectors
        for indices, counts, totals, log_totals in self.docarray:
            # Documents with only one word do us no good.
            doc_len = counts.sum()
            if doc_len < 2:
                continue

            # Sum all word vectors to form one vector for the document.
            doc_vector = hash_vectors[indices].sum(axis=0)

            # Create context vectors for each word by subtracting the
            # vector for that word. Also divide by the document length
            # to prevent long sentencs from having larger effects.
            context_vecs = (doc_vector - hash_vectors[indices]) / doc_len

            # Finally, divide by each word's frequency to prevent common
            # words from having larger effects.
            self.embed_vectors[indices] += context_vecs / totals[:, None]

    def save_vectors(self, filename, mincount=10, truncate=None,
                     include_annotated=True):

        if truncate is None:
            truncate = self.embed_vectors.shape[1]
            embed_vectors = self.embed_vectors
        else:
            embed_vectors = self._eigen_embedding()

        if include_annotated:
            truncate //= 2

        tot = self.docarray._word_count
        wix = self.docarray.word_index
        words = sorted(self.docarray.words,
                       key=lambda w: tot[wix[w]],
                       reverse=True)

        # NOTE: Words ending with '_' are assumed to be "annotated,"
        #       and are merged with their unannotated versions or
        #       dropped from the final output.
        words = [w for w in words
                 if tot[wix[w]] >= mincount and not w.endswith('_')]

        vecs = [embed_vectors[wix[w]][:truncate] for w in words]
        if include_annotated:
            # Concatenate the vectors for annotated and
            # unannotated versions of each word.
            words_ = [w + '_' for w in words]
            vecs_ = [embed_vectors[wix[w_]][:truncate] for w_ in words_]
            vecs = [numpy.concatenate([v, v_])
                    for v, v_ in zip(vecs, vecs_)]

        rows = ['{} {}\n'.format(w, ' '.join(str(v) for v in vec))
                for w, vec in zip(words, vecs)]

        with open(filename, 'w', encoding='utf-8') as op:
            for r in rows:
                op.write(r)

    def save_vocab(self, filename, mincount=10):
        tot = self.docarray._word_count
        wix = self.docarray.word_index
        words = sorted(self.docarray.words,
                       key=lambda w: tot[wix[w]],
                       reverse=True)

        # NOTE: See note above.
        words = [w for w in words
                 if tot[wix[w]] >= mincount and not w.endswith('_')]
        rows = ['{} {}\n'.format(w, tot[wix[w]]) for w in words]
        with open(filename, 'w', encoding='utf-8') as op:
            for r in rows:
                op.write(r)

    def train_test_chunk_sp(self):
        """
        As of 2017-10-22, this passes consistently.
        """
        hash_vectors = self.hash_iter_vectors
        chunksize = 2000
        chunks = ((self.docarray[i:i + chunksize], hash_vectors)
                  for i in range(0, len(self.docarray), chunksize))

        error_records = []
        for ch_ix, chunk in enumerate(chunks):
            r1 = train_chunk_star(chunk)
            r1 = r1[r1.sum(axis=1) > 0]
            r2 = train_chunk_star_sp(chunk)
            r2, r2ix = r2
            try:
                assert numpy.allclose(r1, r2)
            except AssertionError:
                tol = 1e-10
                abs_error = numpy.abs(r1 - r2)
                bad_errors = abs_error[(abs_error > tol).nonzero()]
                bad_error_ratio = bad_errors.shape[0]
                bad_error_ratio /= r1.shape[0] * r1.shape[1]

                print()
                print("Fraction of bad errors (> {}):".format(tol))
                print(bad_error_ratio)
                print("Mean bad error:")
                print(bad_errors.mean())
                print("Median bad error:")
                print(numpy.median(bad_errors))
                print("Max bad error:")
                print(bad_errors.max())
                print()
                if bad_error_ratio > 1e-5:
                    error_records.append((ch_ix, bad_error_ratio))

            self.embed_vectors[r2ix] += r2

        if error_records:
            raise RuntimeError("_train_test_chunk_sp: test failed")
        else:
            print('All chunkwise training tests passed.')

# ###########################
# #### Utility Functions ####
# ###########################

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
    hashes = [list(map(hash, ['{}_{}'.format(w, i) for w in words]))
              for i in range(multiplier)]

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
        return sparsify_rows(hash_arr, sparsifier).astype(numpy.int8)
    else:
        return hash_arr.astype(numpy.int8) * 2 - 1

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

def train_chunk_star(docarray_hash_vectors):
    docarray, hash_vectors = docarray_hash_vectors
    return train_chunk(docarray, hash_vectors)

def train_chunk_star_sp(docarray_hash_vectors):
    docarray, hash_vectors = docarray_hash_vectors
    return train_chunk_cy(docarray, hash_vectors)

def train_chunk_star_jacobian(docarray_hash_vectors):
    docarray, hash_vectors = docarray_hash_vectors
    return train_chunk_jacobian_cy(docarray, hash_vectors)

def train_chunk_star_jac_unnormed(docarray_hash_vectors):
    docarray, hash_vectors = docarray_hash_vectors
    return train_chunk_configurable_cy(docarray,
                                       hash_vectors,
                                       False,
                                       False)

def train_chunk_star_jac_arith(docarray_hash_vectors):
    docarray, hash_vectors = docarray_hash_vectors
    return train_chunk_configurable_cy(docarray,
                                       hash_vectors,
                                       True,
                                       False)

def train_chunk_star_jac_geom(docarray_hash_vectors):
    docarray, hash_vectors = docarray_hash_vectors
    return train_chunk_configurable_cy(docarray,
                                       hash_vectors,
                                       False,
                                       True)

def train_chunk_star_jac_arith_geom(docarray_hash_vectors):
    docarray, hash_vectors = docarray_hash_vectors
    return train_chunk_configurable_cy(docarray,
                                       hash_vectors,
                                       True,
                                       True)



# ########################
# #### CORE FUNCTIONS ####
# ########################

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
