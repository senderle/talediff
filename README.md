# talediff
A prototype word embedding model based on type-level differential operators.

# Concept
This project introduces a new way to reason formally about word embedding
models. Abstractly, suppose for a family of languages `L` we have a vector
space `V` of language features, and a scalar function `T` defined on `V` 
that describes something like the "expressive power" of `L` -- called the 
tale function, "tale" being a word that has long denoted both a *count*
and an *account*. We are interested in the effect of "extending" a member of 
`L` along some dimension `d` of `V`. Concretely, suppose we take a member 
of `L`, `l_1`, and add new words, creating `l_2`. These two languages have 
associated feature vectors in `V` that are equal, except along the dimesion 
`d`, and so `l_2 = l_1 + delta_d`.

Now let us ask: what is the difference between these two languages in terms
of "expressive power"? The fundamental assumtion of this model is that 
the change in expressive power along the changed dimension tells us 
something important about the dimension itself and its relationship to
the language. This amounts to differentiation:

    dT/dV_d = Lim[delta_d -> 0][(T(l_1 + delta_d) - T(l_1)) / delta_d]

Clearly we can do this for any of the dimensions `d` in `V`, and so we 
can imagine a gradient operator that tells something about the way the
tale fuction T behaves in the neighborhood of `l_1`. In particular, it
tells us the direction of steepest ascent for T -- that is, the direction 
in language feature space `V` that points towards the most expressive of 
`l_1`'s immediate neighbors.

Furthermore, we can imagine a *second order* differentiation operator,
which would take the tale function T, and return a function that generates
the Hessian matrix for T in the neighborhood of `l`. This would be the 
*best linear approximation* of the gradient of `T` in the neighborhood of 
`l`. Thus we can use the Hessian to tell us something interesting about 
the local "expressivity landscape" around `l`. 

But what exactly does it tell us? What are the features in `V`? And what
would make a good tale function `T`?

### Word Embedding Vectors Are Type-Level Hessian Matrices

This project takes inspration from McBride 
([2001](http://strictlypositive.org/diff.pdf)) and Abbot et. al. 
([2003](http://strictlypositive.org/derivcont.pdf)), 
who show that algebraic data types are differentiable, and that all the
familiar differential operators from calculus, including the gradient and
Hessian operators, follow essentially identical rules. By formulating
language families as type-level functions that generate composite types
from primitive types, we can construct a range of tale functions, and 
perform exactly the above operations to generate linear approximations of
their behavior in the neighborhoods of particular languages.

To begin with, it is easy to construct a tale function with the property
that its Hessian is just a word-word coocurrence count matrix for a
corpus. First, define a language as the sum type over all possible
sentences in the language. Define a sentence as the product type over
the words that constitute it. And define a word type as a singleton
type. In other words, define a language as a large, high-dimensional
type-level polynomial over singleton word types.

The second derivative of a sentence type defined in the above way is a 
Hessian matrix with zeros for all word pairs not in the sentece at all
(because the derivative of a constant is zero), ones for all hetergeneous
word pairs in the sentence where both of the given words appear just once
in the sentence, and zeros for all homogeneous word "pairs" (diagonals)
where the given word appears just once in the sentence. In (rare) cases
where a word appears twice, there are potentially multiple rules for 
constructing a word-word cooccurrence count matrix; this formalism 
singles out a particular set of rules corresponding to the power rule 
in calculus. (For example, the diagonal entry for a word that appears
twice will be two -- `d/dx x^2 = 2x` evaluated at `x = 1`.)

Thus for each sentence type in the language we can calculate a sparse
Hessian matrix. Now note that differentiation is a linear operator, and
we have defined our language as a sum over sentence types. To find the
Hessian of the langauge, then, we can simply sum up the matrices for all 
the sentences.

As Goldberg and Levy ([2014](https://arxiv.org/abs/1402.3722)) and 
others have shown, many standard word embedding models are essentially
just factorizaitons of this matrix using various weighting and
normalization schemes. Goldberg and Levy ask "Why does this produce good
word representations?" The above construction provides a new answer to
that question, at least if we assume that "good" is an underspecified 
synonym for "linear":

*The word-word coocurrence matrix is a linear operator that provides a good
local approximation of the gradient of a function describing a language's 
expressive power with respect to the words in the language.*

### Deriving Novel Embedding and Reduction Schemes

In this word embedding model, the linearity of induced semantic structures
directly emerges from the linearity of the differential operator. This is
guaranteed by the logic of differentiation over types, which produces,
in McBride's words, "one-hole contexts." That is, type-level differentiation
gives us a way to extract linear structures from context types.

Given that linearity is a fundamental attribute of the word-word coocurrence
matrix, we can expect many different dimension-reduction schemes to work
reasonably well -- even random projection! Furthermore, we can expect that
more complex sentence types, when they can be represented as algebriac 
data types, will admit methods similar to the above. For example, this
approach should allow tree structures (branching recursive types) to be 
analyzed as easily as lists (unicursal recursive types). And finally, we can
expect that choosing non-singleton word types will yield new and potentially
useful results.

The code in this repository is under active development, and takes a few
very small steps in the above directions. 
