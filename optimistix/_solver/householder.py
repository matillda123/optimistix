import functools as ft
from collections.abc import Callable
from typing import Any, ClassVar, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PyTree, Scalar

from .newton_chord import _NoAux

from .._custom_types import Aux, Fn
from .._root_find import AbstractRootFinder
from .._solution import RESULTS
from .._search import (
    FunctionInfo,
)




class _HouseholderState(eqx.Module):
    y_prev: Scalar
    f_info: FunctionInfo.Eval
    householder_update: Callable



def _inverse_function(fn):
    def wrapper(*args, **kwargs):
        val = fn(*args, **kwargs)
        return 1/val
    return wrapper


def _get_nth_derivative(fn, n):
    for _ in range(n):
        fn = jax.grad(fn, argnums=0)
    return fn


def _get_householder_update(fn, n):
    fn_inv = _inverse_function(fn)
    fn_inv_n1prime = _get_nth_derivative(fn_inv, n-1)
    fn_inv_nprime = _get_nth_derivative(fn_inv, n)
    #fn_inv_nprime = jax.grad(fn_inv_n1prime) # not sure which is more efficient.

    def wrapper(*args, **kwargs):
        numerator = fn_inv_n1prime(*args, **kwargs)
        denominator = fn_inv_nprime(*args, **kwargs)
        return n*numerator/denominator
    
    return jax.jit(wrapper)
    #return wrapper





class Householder(AbstractRootFinder[Scalar, Scalar, Aux, _HouseholderState]):
    """Householder's method of root finding. A general class of Newton-type methods. 
    (e.g. order=1 is the Newton-Raphson method, order=2 is Halley's method, ...) 
    Even though the convergence rate equals the order+1, the increased computational demand for the evaluation 
    of the n-th derivative makes higher order methods barely usable. Additionally, higher order methods do 
    not exhibit higher stability. They suffer from the same divergence issues as Newton-Raphson.
    
    This may only be used with functions `R->R`, i.e. functions with scalar input and scalar output.
    """

    order: int
    rtol: float
    atol: float
    # All norms are the same for scalars.
    norm: ClassVar[Callable[[PyTree], Scalar]] = jnp.abs

    def init(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _HouseholderState:
        
        f_eval, aux = fn(y, args)

        if y.ndim > 0:
            raise ValueError(
                "Householder can only be used to find the roots of a function taking a "
                "scalar input. 1x1-Arrays are also not permitted."
            )

        if f_eval.ndim > 0:
            raise ValueError(
                "Householder can only be used to find the roots of a function producing a "
                "scalar input. 1x1-Arrays are also not permitted."
            )

        householder_update = _get_householder_update(_NoAux(fn), self.order)

        return _HouseholderState(
            y_prev=y,
            f_info=FunctionInfo.Eval(f_eval),
            householder_update = householder_update
        )

    def step(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: _HouseholderState,
        tags: frozenset[object],
    ) -> tuple[Scalar, _HouseholderState, Aux]:

        new_y = y + state.householder_update(y, args)
        f_eval, aux = fn(new_y, args)
       
        new_state = _HouseholderState(
            y_prev=y,
            f_info=FunctionInfo.Eval(f_eval),
            householder_update = state.householder_update
        )
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: _HouseholderState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        del fn, args, options
        scale = self.atol + self.rtol * jnp.abs(y)
        y_small = jnp.abs(y - state.y_prev) < scale
        f_small = jnp.abs(state.f_info.f) < self.atol
        # if y=Nan, the denominator in housholder update was probably zero, the solve was likely successful in this case
        # however a warning should be printed. But idk how to do that with jax.jit.
        return ((y_small & f_small) | jnp.isnan(y)), RESULTS.successful

    def postprocess(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _HouseholderState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Scalar, Aux, dict[str, Any]]:
            
        y = jnp.where(jnp.isnan(y), state.y_prev, y)
        return y, aux, {}


