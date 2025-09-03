from collections.abc import Callable
from typing import Any, Generic, TYPE_CHECKING

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

# the semipositive definite tag in _identity_pytree is wrong, but unused. So its fine to import it.
from .quasi_newton import _outer, _identity_pytree

from .._custom_types import Aux, Fn, Out, Y
from .._misc import cauchy_termination, max_norm, tree_dtype, tree_full_like, tree_dot
from .._root_find import AbstractRootFinder
from .._solution import RESULTS


from .._search import FunctionInfo









class _BroydenState(eqx.Module):
    jinvprev: Y
    diff_y: Y
    f_info: FunctionInfo.Eval
    result: RESULTS



def update_jacinv_good_broyden(jprev, dy, df):
    p = (dy**ω - jprev.mv(df)**ω).ω
    q = tree_dot(dy, jprev.mv(df))
    w = jprev.transpose().mv(dy) # is this the same as dy @ jprev? im not sure.

    j = jprev.pytree
    pw = _outer(p, w)
    pw = jax.tree.map(lambda leaf: leaf/q, pw)
    j = (j**ω + pw**ω).ω
    return lx.PyTreeLinearOperator(j, jax.eval_shape(lambda: df))



def update_jacinv_bad_broyden(jprev, dy, df):
    p = (dy**ω - jprev.mv(df)**ω).ω
    q = tree_dot(df, df)
    w = df

    j = jprev.pytree
    pw = _outer(p, w)
    pw = jax.tree.map(lambda leaf: leaf/q, pw)
    j = (j**ω + pw**ω).ω
    return lx.PyTreeLinearOperator(j, jax.eval_shape(lambda: df))



class _AbstractBroyden(AbstractRootFinder[Y, Out, Aux, _BroydenState]):

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm
    _is_good_broyden: AbstractClassVar[bool]


    def init(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _BroydenState:
        
        # starting with the actual jacobian inverse or an approximate would be better
        jinvprev = _identity_pytree(y) # input should be y(?), because J^-1 maps from f-space to y-space. 
        f_eval, aux = fn(y, args)
        diff_y = jax.tree.map(lambda leaf: jnp.full_like(leaf, jnp.inf), y)

        return _BroydenState(
            jinvprev = jinvprev,
            diff_y = diff_y,
            f_info = FunctionInfo.Eval(f_eval),
            result=RESULTS.successful,
        )

    def step(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _BroydenState,
        tags: frozenset[object],
    ) -> tuple[Y, _BroydenState, Aux]:
        lower = options.get("lower")
        upper = options.get("upper")
        del options

        jinvprev, fprev = state.jinvprev, state.f_info.f
        diff_y = (-1*jinvprev.mv(fprev)**ω).ω
        new_y = (y**ω + diff_y**ω).ω
        
        if lower is not None:
            new_y = jtu.tree_map(lambda a, b: jnp.clip(a, min=b), new_y, lower)
        if upper is not None:
            new_y = jtu.tree_map(lambda a, b: jnp.clip(a, max=b), new_y, upper)
        if lower is not None or upper is not None:
            diff_y = (y**ω - new_y**ω).ω

        f_eval, aux = fn(new_y, args)
        diff_f = (f_eval**ω - fprev**ω).ω

        if self._is_good_broyden:
            jinvprev = update_jacinv_good_broyden(jinvprev, diff_y, diff_f)
        else:
            jinvprev = update_jacinv_bad_broyden(jinvprev, diff_y, diff_f)

        new_state = _BroydenState(
            jinvprev = jinvprev,
            diff_y = diff_y,
            f_info = FunctionInfo.Eval(f_eval),
            result=RESULTS.successful,
        )
        return new_y, new_state, aux
    

    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _BroydenState,
        tags: frozenset[object],
    ):
        del fn, args, options
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
        return terminate, terminate_result

    def postprocess(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _BroydenState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}








class Broyden(_AbstractBroyden[Y, Out, Aux]):
    """Broyden's "good" method for root finding.

    Works similar to Newton's method. Instead of computing the exact Jacobian, the method updates an 
    approximate inverse Jacobian each iteration. This is somewhat analogous to the BFGS-Method for minimization. 


    This solver optionally accepts the following `options`:

    - `lower`: The lower bound on the hypercube which contains the root. The
        iterates of `y` will be clipped to this hypercube.
    - `upper`: The upper bound on the hypercube which contains the root. The
        iterates of `y` will be clipped to this hypercube.
    """

    _is_good_broyden = True







class BadBroyden(_AbstractBroyden[Y, Out, Aux]):
    """Broyden's "bad" method for root finding.

    Works similar to Newton's method. Instead of computing the exact Jacobian, the method updates an 
    approximate inverse Jacobian each iteration. The method has the byname "bad" because it has been 
    observed to perform worse than Broyden's good method. However apparently this depends on the circumstances. 
    Broyden's methods differ in the formulas for the update of the inverse Jacobian approximation.


    This solver optionally accepts the following `options`:

    - `lower`: The lower bound on the hypercube which contains the root. The
        iterates of `y` will be clipped to this hypercube.
    - `upper`: The upper bound on the hypercube which contains the root. The
        iterates of `y` will be clipped to this hypercube.
    """

    _is_good_broyden = False