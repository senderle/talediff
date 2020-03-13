import numba
import numpy
import timeit

from time import sleep
from math import log, exp

_return_type=numba.types.Tuple((numba.float64,
                                numba.float64[:],
                                numba.float64[:, :]))
@numba.jit(_return_type(numba.float64[:], numba.float64[:]), 
           nopython=True,
           nogil=True)
def simple_hessian(ps, xs):
    f_out = numpy.prod(numpy.power(xs, ps))
    jac = numpy.empty((len(ps),))
    jac[:] = f_out
    hess = numpy.empty((len(ps), len(ps)))

    for i in range(len(ps)):
        jac[i] *= ps[i] / xs[i]
        for j in range(len(ps)):
            ps_j = ps[j] if i != j else ps[j] - 1
            hess[i, j] = jac[i] * ps_j / xs[j]

    return f_out, jac, hess

_return_type=numba.types.Tuple((numba.float64,
                                numba.float64[:],
                                numba.float64[:, :]))
@numba.jit(_return_type(numba.float64[:], numba.float64[:]), 
           nopython=True,
           nogil=True)
def simple_log_hessian(ps, xs):
    log_ps = numpy.log(ps)
    log_xs = numpy.log(xs)
    f_out = (log_xs * ps).sum()
    jac = numpy.empty((len(ps),))
    jac[:] = f_out
    hess = numpy.empty((len(ps), len(ps)))

    for i in range(len(ps)):
        jac[i] += log_ps[i] - log_xs[i]
        for j in range(len(ps)):
            ps_j = ps[j] if i != j else ps[j] - 1
            hess[i, j] = exp(jac[i] - log_xs[j]) * ps_j

    return exp(f_out), numpy.exp(jac), hess

@numba.jit(numba.void(numba.float64[:],
                      numba.float64[:, :],
                      numba.float64[:, :, :],
                      numba.int64[:],
                      numba.float64[:],
                      numba.float64[:]),
           nopython=True,
           nogil=True,
           parallel=True)
def simple_hessian_multi(f_out,
                         jacobian_out,
                         hessian_out,
                         ends,
                         counts,
                         weights):
    n_docs = ends.shape[0]
    starts = numpy.empty(ends.shape, dtype=numpy.int64)
    starts[0] = 0
    starts[1:] = ends[:-1]
    for doc_ix in numba.prange(n_docs):
        start = starts[doc_ix]
        end = ends[doc_ix]
        doc_counts = counts[start: end]
        doc_weights = weights[start: end]
        f, jac, hess = simple_hessian(doc_counts, doc_weights)
        f_out[doc_ix] = f
        jacobian_out[doc_ix] = jac
        hessian_out[doc_ix] = hess

def _hessian_test_wrapper_multi(PS, XS):
    n_docs, n_dims = PS.shape
    f_out = numpy.zeros((n_docs,), dtype=numpy.float64)
    jacobian_out = numpy.zeros((n_docs, n_dims), dtype=numpy.float64)
    hessian_out = numpy.zeros((n_docs, n_dims, n_dims), dtype=numpy.float64)
    ends = numpy.arange(0, n_dims * n_docs, n_dims) + n_dims
    counts = numpy.array(PS.ravel(), dtype=numpy.float64)
    weights = numpy.array(XS.ravel(), dtype=numpy.float64)
    simple_hessian_multi(f_out, jacobian_out, hessian_out, ends, counts, weights)
    return hessian_out[0]

def _estimated_hessian(ps, xs, delta=1e-4):
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

def _test_hessian_speed():
    setup = """
from sparsehessnumba import (  # This module!
    simple_hessian, 
    simple_log_hessian,
    _hessian_test_wrapper_multi
)
import numpy 

n_vars = 50
n_tests = 100000
PS = numpy.random.randint(1, 8, size=(n_tests, n_vars))
PS = numpy.asarray(PS, dtype=numpy.float64)
XS = numpy.random.random((n_tests, n_vars)) * 3
    """

    stmt_template = """
for ps, xs in zip(PS, XS):
    {}(ps, xs)
    """
    
    funcs = []  # ['simple_hessian', 'simple_log_hessian']
    print()
    print('Hessian function performance tests:')
    for f in funcs:
        print()
        print(f'Timing {f}')
        stmt = stmt_template.format(f)
        timeit.main(['-s', setup, stmt])

    print()
    print('Timing _hessian_test_wrapper_multi')
    stmt = """
_hessian_test_wrapper_multi(PS, XS)
"""
    timeit.main(['-s', setup, stmt])

def _test_hessian_correctness():
    n_vars = 50
    n_tests = 100
    PS = numpy.random.randint(1, 8, size=(n_tests, n_vars))
    PS = numpy.asarray(PS, dtype=numpy.float64)
    XS = numpy.random.random((n_tests, n_vars)) * 2 + 1
    delta = 1e-6

    failed = 0
    est_failed = 0
    for ps, xs in zip(PS, XS):
        _f, _j, simple_hessian_result = simple_hessian(ps, xs)
        _f, _j, simple_log_hessian_result = simple_log_hessian(ps, xs)
        est_result = _estimated_hessian(ps, xs)
        test_versions = ((simple_hessian_result, "simple_hessian"),
                         (simple_log_hessian_result, "simple_log_hessian"))
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
                    if worst_errors.any() and res_name != "estimated_hessian":
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
        print("All hessian tests passed.")

if __name__ == '__main__':
    _test_hessian_correctness()
    _test_hessian_speed()
