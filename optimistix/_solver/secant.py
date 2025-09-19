from collections.abc import Callable
from typing import Any, Generic, TYPE_CHECKING, List

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .newton_chord import _small, _converged, _diverged
from .quasi_newton import _outer, _identity_pytree

from .._custom_types import Aux, Fn, Out, Y
from .._misc import cauchy_termination, max_norm, tree_dtype, tree_full_like, tree_dot
from .._root_find import AbstractRootFinder
from .._solution import RESULTS


from .._search import FunctionInfo




# def _orthonormal_basis_for_pytree(pytree: PyTree[Array]) -> list[PyTree[Array]]:
#     """Creates a list with orthonormal basis "vectors" for a given pytree shape, 
#     where the individual leafs are treated as dimensions. Such that e.g. tree_dot(basis_i, basis_j)=δ_ij.

#     **Arguments**:

#     - `pytree`: A pytree such that the output of `_orthonormal_basis_for_pytree` is a list of orthonormal 
#     pytrees of the same structure as `pytree`.

#     **Returns**:
#     A list of basis-pytrees which span the current pytree-space.
#     """
#     leaves, structure = jtu.tree_flatten(pytree)
#     pytree_basis = []
#     for N_basis in range(len(leaves)):
#         basis_leaves = []
#         for i1, l1 in enumerate(leaves):
#             if i1 == N_basis:
#                 arr = jnp.ones(jnp.shape(l1))
#                 basis_leaves.append(arr/jnp.sqrt(jnp.size(arr)))
#             else:
#                 arr = jnp.zeros(jnp.shape(l1))
#                 basis_leaves.append(arr)
#         pytree_basis.append(jtu.tree_unflatten(structure, basis_leaves))
#     return pytree_basis



def _get_leaf_with_one(i, start, zero_leaf):
    local_idx = i - start
    leaf = zero_leaf.flatten()
    leaf = leaf.at[local_idx].set(1.0)
    leaf = leaf.reshape(jnp.shape(zero_leaf))
    return leaf


def _get_leaf_with_zeros(i, start, zero_leaf):
    return zero_leaf



def _orthonormal_basis_for_pytree(pytree: PyTree[Array]) -> list[PyTree[Array]]:
    """Creates a list with orthonormal basis "vectors" for a given pytree shape.

     **Arguments**:

     - `pytree`: A pytree such that the output of `_orthonormal_basis_for_pytree` is a list of orthonormal 
     pytrees of the same structure as `pytree`.

     **Returns**:
     A list of basis-pytrees which span the current pytree-space.
     """
    
    leaves, treedef = jax.tree.flatten(pytree)

    flat_elements = jnp.concatenate([leaf.flatten() for leaf in leaves])
    n_elements = len(flat_elements)

    zero_leaves = [0*leaf for leaf in leaves]

    leaf_sizes = [leaf.size for leaf in leaves]
    cumulative_sizes = jnp.cumsum(jnp.array([0] + leaf_sizes))

    pytree_basis = []
    for i in range(n_elements):
        basis_leaves = []
        for j, (start, end) in enumerate(zip(cumulative_sizes[:-1], cumulative_sizes[1:])):
            get_one = ((start <= i ) & (i < end))
            leaf = jax.lax.cond(get_one, 
                                _get_leaf_with_one, 
                                _get_leaf_with_zeros, 
                                i, start, zero_leaves[j])
            basis_leaves.append(leaf)

        pytree_basis.append(jax.tree.unflatten(treedef, basis_leaves))
    return pytree_basis





def _build_J_approx_naive(J_approx, f_vals, pytree_basis):
    '''
    Assembles a Jacobian approximate by addition of outer products between f_vals[i] and pytree_basis[i]. 
    Is inefficient because pytree_basis is mostly zeros
    '''

    for f_val, basis in zip(f_vals, pytree_basis):
        outer = _outer(f_val, basis)
        J_approx = (J_approx**ω + outer**ω).ω
    return J_approx




def _build_J_approx(f_vals, y):
    '''
    Assembles an approximate jacobian by stacking f_vals into the structure of the jacobian. 

    If y and f are both 1D, J is 2D. In such a case one would simply use jnp.stack. 
    This function generalizes this to arbitrary pytrees. (i hope)
    
    '''

    leaves_f0, structure_f = jax.tree.flatten(f_vals[0])
    leaves_y, structure_y = jax.tree.flatten(y)
    structure_J = structure_f.compose(structure_y) # get treedef of jacobian

    leaves_f = [jax.tree.flatten(f_val)[0] for f_val in f_vals] # get all f_vals values 
    leaves_f = list(map(list, zip(*leaves_f))) # transpose leaves_f, (N_basis, leaves_one_f) -> (leaves_one_f, N_basis)

    leaf_sizes = [leaf.size for leaf in leaves_y]
    cumulative_sizes = jnp.cumsum(jnp.array([0] + leaf_sizes)) # get size of each leaf in y

    # iterate through leaves of one f_val
    # get all f_val leaves that originate from one leaf of y
    # jnp.asarray essentially stacks them
    # finally they are reshaped into the correct shape
    leaves_J = [
        jnp.roll(jnp.asarray(leaves_f[j]), -1*cumulative_sizes[i])[0:leaf_sizes[i]].reshape(jnp.shape(leaves_f[j][i]) + jnp.shape(leaves_y[i]), order="F") 
        for i in range(len(cumulative_sizes)-1)
        for j in range(len(leaves_f))
        ]
    
    return jax.tree.unflatten(structure_J, leaves_J)




class _SecantState(eqx.Module, Generic[Y]):
    y1: Y
    I: Y
    basis: list[Y]
    diff_y: Y
    J_init: Y
    f_info: FunctionInfo.Eval
    result: RESULTS
    diffsize: Scalar
    diffsize_prev: Scalar
    step: Scalar





class _AbstractSecant(AbstractRootFinder[Y, Out, Aux, _SecantState]):

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm
    kappa: float = 1e-2
    linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
    cauchy_termination: bool = True
    
    _is_Secant: AbstractClassVar[bool]


    def init(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _SecantState:
        
        # idk how to handle weak_type differences between y0 and y1, so i came up with this
        y1 = jax.tree.map(lambda x: jnp.asarray(x), options.get("y1"))
        y = jax.tree.map(lambda x: jnp.asarray(x, dtype=x.dtype), y)
        y1 = jax.tree.map(lambda x: jnp.asarray(x, dtype=x.dtype), y1)

        if jax.eval_shape(lambda: y)!=jax.eval_shape(lambda: y1):
            raise ValueError(
                "y0 and y1 need to have the same structure/shape"
            )

        I = _identity_pytree(y)
        pytree_basis = _orthonormal_basis_for_pytree(y)
        
        f_eval, aux = fn(y, args)

        # creates a pytree with all zeros in the shape of the jacobian
        J_init = _outer(f_eval, pytree_basis[0])
        J_init = jax.tree.map(lambda leaf: 0*leaf, J_init)

        diff_y = (y**ω - y1**ω).ω
        dtype = tree_dtype(f_struct)
        return _SecantState(
            y1 = y1,
            I = I,
            basis = pytree_basis,
            J_init = J_init,
            diff_y=diff_y,
            f_info = FunctionInfo.Eval(f_eval),
            result=RESULTS.successful,
            diffsize=jnp.array(jnp.inf, dtype=dtype),
            diffsize_prev=jnp.array(1.0, dtype=dtype),
            step=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _SecantState,
        tags: frozenset[object],
    ) -> tuple[Y, _SecantState, Aux]:
        lower = options.get("lower")
        upper = options.get("upper")
        del options

        y1, diff_y, I, pytree_basis, J_approx = state.y1, state.diff_y, state.I, state.basis, state.J_init
        diff_y_norm = jnp.sqrt(tree_dot(diff_y, diff_y))

        f_eval, aux = fn(y, args)
        f_vals = [fn((y**ω + diff_y_norm*I.mv(basis)**ω).ω, args)[0] for basis in pytree_basis]
        f_vals = [(f_val**ω - f_eval**ω).ω for f_val in f_vals]

        #J_approx = _build_J_approx_naive(J_approx, f_vals, pytree_basis)
        J_approx = _build_J_approx(f_vals, y)
        J_approx = ((1/diff_y_norm)*J_approx**ω).ω

        J = lx.PyTreeLinearOperator(J_approx, output_structure=jax.eval_shape(lambda: f_eval))
        sol = lx.linear_solve(J, f_eval, solver=self.linear_solver, throw=False)

        new_y = (y**ω - sol.value**ω).ω

        if lower is not None:
            new_y = jtu.tree_map(lambda a, b: jnp.clip(a, min=b), new_y, lower)
        if upper is not None:
            new_y = jtu.tree_map(lambda a, b: jnp.clip(a, max=b), new_y, upper)
        if lower is not None or upper is not None:
            diff_y = (y**ω - new_y**ω).ω
        else:
            diff_y = (-1*sol.value**ω).ω

        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        with jax.numpy_dtype_promotion("standard"):
            diffsize = self.norm((diff_y**ω / scale**ω).ω)

        if self._is_Secant:
            y1 = y
        else:
            y1 = y1

        new_state = _SecantState(
            y1 = y1,
            I = I,
            basis = pytree_basis,
            diff_y = diff_y,
            J_init = state.J_init,
            f_info = FunctionInfo.Eval(f_eval),
            diffsize = jnp.asarray(diffsize, dtype=state.diffsize.dtype),
            diffsize_prev = state.diffsize,
            result = RESULTS.promote(sol.result),
            step = state.step + 1,
        )
        return new_y, new_state, aux
    

    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _SecantState[Y],
        tags: frozenset[object],
    ):
        del fn, args, options
        if self.cauchy_termination:
            # Compare `f_val` against 0, not against some `f_prev`. This is because
            # we're doing a root-find and know that we're aiming to get close to zero.
            # Note that this does mean that the `rtol` is ignored in f-space, and only
            # `atol` matters.
            terminate = cauchy_termination(
                self.rtol,
                self.atol,
                self.norm,
                y,
                state.diff_y,
                jtu.tree_map(jnp.zeros_like, state.f_info.f),
                state.f_info.f,
            )
            terminate_result = RESULTS.successful
        else:
            # TODO(kidger): perform only one iteration when solving a linear system!
            at_least_two = state.step >= 2
            rate = state.diffsize / state.diffsize_prev
            factor = state.diffsize * rate / (1 - rate)
            small = _small(state.diffsize)
            diverged = _diverged(rate)
            converged = _converged(factor, self.kappa)
            terminate = at_least_two & (small | diverged | converged)
            terminate_result = RESULTS.where(
                jnp.invert(small) & (diverged | jnp.invert(converged)),
                RESULTS.nonlinear_divergence,
                RESULTS.successful,
            )
        linsolve_fail = state.result != RESULTS.successful
        result = RESULTS.where(linsolve_fail, state.result, terminate_result)
        terminate = linsolve_fail | terminate
        return terminate, result


    def postprocess(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _SecantState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}








class Secant(_AbstractSecant[Y, Out, Aux]):
    """A multivariate version of the Secant method. Developed by S. Robinson (https://epubs.siam.org/doi/abs/10.1137/0703057).
    Each iteration a new approximation of the Jacobian is constructed based on the location and corresponding function values of 
    two points. This jacobian is the used as in the Newton-Raphson method. 
    However in addition to the linear solver the method requires N function evaluations per iteration, where N is the dimensionality 
    of the function input.


    This solver requires the following `options`:

    - `y1`: An additional initial guess with the same shape/structure as y0.

    
    This solver optionally accepts the following `options`:

    - `lower`: The lower bound on the hypercube which contains the root. The
        iterates of `y` will be clipped to this hypercube.
    - `upper`: The upper bound on the hypercube which contains the root. The
        iterates of `y` will be clipped to this hypercube.
    """

    _is_Secant = True







class FalsePosition(_AbstractSecant[Y, Out, Aux]):
    """A multivariate version of the False-Position method also known as Regula-Falsi method. This implementation is 
    based on the multivariate Secant method of S. Robinson (https://epubs.siam.org/doi/abs/10.1137/0703057). The method differs 
    from the multivariate Secant method in the fact that the second initial guess `y1` is held fixed throughout the entire optimization. 

    
    This solver requires the following `options`:

    - `y1`: An additional initial guess with the same shape/structure as y0.

    
    This solver optionally accepts the following `options`:

    - `lower`: The lower bound on the hypercube which contains the root. The
        iterates of `y` will be clipped to this hypercube.
    - `upper`: The upper bound on the hypercube which contains the root. The
        iterates of `y` will be clipped to this hypercube.
    """

    _is_Secant = False