from collections.abc import Callable
from typing import Any, Generic, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import Partial, AbstractVar
from equinox.internal import ω
from jaxtyping import Array, Bool, PyTree, Scalar

from .._custom_types import Aux, DescentState, Fn, Out, SearchState, Y
from .._minimise import AbstractMinimiser
from .._misc import (
    cauchy_termination,
    max_norm,
)
from .._search import (
    FunctionInfo,
)
from .._solution import RESULTS






class _GoldenSectionState(eqx.Module):
    lower: float
    upper: float
    y1: float
    y2: float
    f1: float
    f2: float

    first_step: Bool[Array, ""]
    f_info: FunctionInfo.Eval
    
    terminate: Bool[Array, ""]
    result: RESULTS



def _move_towards_lower(lower, upper, y1, y2, f1, f2, fn, args):
    phi = jnp.array(1.618034)
    lower, upper, y2 = lower, y2, y1
    y1 = upper + (lower - upper)/phi
    f2 = f1
    f1, aux1 = fn(y1, args)
    return jnp.array([lower, upper, y1, y2, f1, f2])


def _move_towards_upper(lower, upper, y1, y2, f1, f2, fn, args):
    phi = jnp.array(1.618034)
    lower, upper, y1 = y1, upper, y2
    y2 = lower + (upper-lower)/phi
    f1 = f2
    f2, aux2 = fn(y2, args)
    return jnp.array([lower, upper, y1, y2, f1, f2])



class GoldenSectionSearch(AbstractMinimiser[Y, Aux, _GoldenSectionState]):
    """Golden Section Search for minimisation of 1D functions.
    An interval containing a minimum is subdivided using to the golden ratio to produce two guesses `[y1, y2]` for the location of the minimum. 
    Based on the function values at these points the interval is shrunk down by setting one of `[y1, y2]` as the new boundary. 
    This approach reduces the uncertainty by a factor of 0.618 each iteration.


    Requires the following `options`:

    - `lower`: The lower bound on the interval which contains the minimum.
    - `upper`: The upper bound on the interval which contains the minimum.

    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _GoldenSectionState:
        
        phi = jnp.array(1.618034)
        lower, upper = jnp.array(options.get("lower"), dtype=float), jnp.array(options.get("upper"), dtype=float)

        if jnp.shape(y) != () or jnp.shape(lower) != () or jnp.shape(upper) != ():
            raise ValueError(
                "Golden-Section-Search can only be used to find the minima of a function taking a "
                "scalar input."
            )
        if not isinstance(f_struct, jax.ShapeDtypeStruct) or f_struct.shape != ():
            raise ValueError(
                "Golden-Section-Search can only be used to find the minima of a function producing a "
                "scalar input."
            )

        y1 = upper + (lower-upper)/phi
        y2 = lower + (upper-lower)/phi

        f1, aux1 = fn(y1, args)
        f2, aux2 = fn(y2, args)
        f_eval, aux_eval = fn(y, args)

        return _GoldenSectionState(
            lower=lower,
            upper=upper,
            y1=y1,
            y2=y2,
            f1=f1,
            f2=f2,
            f_info=FunctionInfo.Eval(f_eval),
            first_step=jnp.array(True),
            terminate=jnp.array(False),
            result=RESULTS.successful,
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _GoldenSectionState,
        tags: frozenset[object],
    ) -> tuple[Y, _GoldenSectionState, Aux]:
        
        lower, upper, y1, y2, f1, f2 = state.lower, state.upper, state.y1, state.y2, state.f1, state.f2

        move_towards_lower = (f1 < f2)
        lower, upper, y1, y2, f1, f2 = jnp.where(move_towards_lower, 
                                                 _move_towards_lower(lower, upper, y1, y2, f1, f2, fn, args), 
                                                 _move_towards_upper(lower, upper, y1, y2, f1, f2, fn, args))
        
        # towards_lower = Partial(_move_towards_lower, fn=fn, args=args)
        # towards_upper = Partial(_move_towards_upper, fn=fn, args=args)
        # lower, upper, y1, y2, f1, f2 = jax.lax.cond(move_towards_lower, 
        #                                             towards_lower, 
        #                                             towards_upper, 
        #                                             lower, upper, y1, y2, f1, f2)
        
        # this doesnt have to be done every iteration in GSS
        y_eval = (y1 + y2)/2
        f_eval, aux_eval = fn(y_eval, args)
        f_eval_info = FunctionInfo.Eval(f_eval)
        

        def accepted():
            y_diff = (y_eval**ω - y**ω).ω
            f_diff = (f_eval**ω - state.f_info.f**ω).ω
            terminate = cauchy_termination(
                self.rtol, self.atol, self.norm, y_eval, y_diff, f_eval, f_diff
            )
            terminate = jnp.where(
                state.first_step, jnp.array(False), terminate
            )  # Skip termination on first step
            return y_eval, f_eval_info, aux_eval, terminate

        y, f_info, aux, terminate = accepted()
        result = RESULTS.successful

        state = _GoldenSectionState(
            lower=lower,
            upper=upper,
            y1=y1,
            y2=y2,
            f1=f1,
            f2=f2,
            f_info=f_info,
            first_step=jnp.array(False),
            terminate=terminate,
            result=result,
        )
        return y, state, aux
    


    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _GoldenSectionState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _GoldenSectionState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}
    














class _JarrattState(eqx.Module):
    y: float
    y1: float
    y2: float
    f1: float
    f2: float
    
    first_step: Bool[Array, ""]
    f_info: FunctionInfo.Eval
    
    terminate: Bool[Array, ""]
    result: RESULTS



class Jarratt(AbstractMinimiser[Y, Aux, _JarrattState]):
    """Jarratt's method for minimization of 1D-functions. Also known as Successive Parabolic Interpolation (SPI). 
    Each iteration a parabola is fitted through the current and previous two guesses. The location of the extremum of this parabola 
    is used as the updated guess of the minimum. 
    The method is not guaranteed to find a minimum. It may also find maxima or saddle points instead. Additionally the method can diverge. 

    
    Requires the following `options`:
    - `y1`: A previous guess for the location of the minimum.
    - `y2`: A previous guess for the location of the minimum.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _JarrattState:
        
        y1, y2 = jnp.array(options.get("y1"), dtype=float), jnp.array(options.get("y2"), dtype=float)

        if jnp.shape(y) != () or jnp.shape(y1) != () or jnp.shape(y2) != ():
            raise ValueError(
                "Jarratt can only be used to find the minima of a function taking a "
                "scalar input."
            )
        if not isinstance(f_struct, jax.ShapeDtypeStruct) or f_struct.shape != ():
            raise ValueError(
                "Jarratt can only be used to find the minima of a function producing a "
                "scalar input."
            )



        f1, aux1 = fn(y1, args)
        f2, aux2 = fn(y2, args)
        f_eval, aux_eval = fn(y, args)
        
        
        return _JarrattState(
            y=y,
            y1=y1,
            y2=y2,
            f1=f1,
            f2=f2,
            f_info=FunctionInfo.Eval(f_eval),
            first_step=jnp.array(True),
            terminate=jnp.array(False),
            result=RESULTS.successful,
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _JarrattState,
        tags: frozenset[object],
    ) -> tuple[Y, _JarrattState, Aux]:
        y1, y2, f1, f2 = state.y1, state.y2, state.f1, state.f2
        f = state.f_info.f

        p = (y1-y)**2*(f-f2)+(y2-y)**2*(f1-f)
        q = (y1-y)*(f-f2) + (y2-y)*(f1-f)
        y_eval = y + 0.5*p/(q + 1e-15)
        
        f_eval, aux_eval = fn(y_eval, args)
        f_eval_info = FunctionInfo.Eval(f_eval)

        def accepted():
            y_diff = (y_eval**ω - y**ω).ω
            f_diff = (f_eval**ω - state.f_info.f**ω).ω
            terminate = cauchy_termination(
                self.rtol, self.atol, self.norm, y_eval, y_diff, f_eval, f_diff
            )
            terminate = jnp.where(
                state.first_step, jnp.array(False), terminate
            )  # Skip termination on first step
            return y_eval, f_eval_info, aux_eval, terminate

        y, f_info, aux, terminate = accepted()
        result = RESULTS.successful

        state = _JarrattState(
            y=y,
            y1=y1,
            y2=y2,
            f1=f1,
            f2=f2,
            f_info=f_info,
            first_step=jnp.array(False),
            terminate=terminate,
            result=result,
        )
        return y, state, aux
    


    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _JarrattState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _JarrattState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}
    


























class _BrentState(eqx.Module):
    y1: float
    y2: float
    f1: float
    f2: float

    jarrat_quotient: float
    
    first_step: Bool[Array, ""]
    f_info: FunctionInfo.Eval
    
    terminate: Bool[Array, ""]
    result: RESULTS




class Brent(AbstractMinimiser[Y, Aux, _BrentState]):
    """Brent's method for minimization of 1D functions. 
    Analogously to the Brent-Dekker method in root-finding, this algorithm combines two algorithms (Golden-Section-Search and Jarratt's method) 
    in order to obtain guaranteed convergence with a superlinear convergence rate. Each iteration the algorithm attempts Jarratt's method. 
    If this fails the algorithm falls back to Golden-Section-Search.

    
    Requires the following `options`:

    - `lower`: The lower bound on the interval which contains the minimum.
    - `upper`: The upper bound on the interval which contains the minimum.

    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar] = max_norm

    def init(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: jax.ShapeDtypeStruct,
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _JarrattState:
        
        y1, y2 = jnp.asarray(options.get("lower"), dtype=float), jnp.asarray(options.get("upper"), dtype=float)

        if jnp.shape(y) != () or jnp.shape(y1) != () or jnp.shape(y2) != ():
            raise ValueError(
                "Brent can only be used to find the minima of a function taking a "
                "scalar input."
            )
        if not isinstance(f_struct, jax.ShapeDtypeStruct) or f_struct.shape != ():
            raise ValueError(
                "Brent can only be used to find the minima of a function producing a "
                "scalar input."
            )

        f1, aux1 = fn(y1, args)
        f2, aux2 = fn(y2, args)
        f_eval, aux_eval = fn(y, args)
        
        
        return _BrentState(
            y1=y1,
            y2=y2,
            f1=f1,
            f2=f2,
            jarrat_quotient=jnp.array(1.0),
            f_info=FunctionInfo.Eval(f_eval),
            first_step=jnp.array(True),
            terminate=jnp.array(False),
            result=RESULTS.successful,
        )

    def step(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _JarrattState,
        tags: frozenset[object],
    ) -> tuple[Y, _JarrattState, Aux]:
        
        phi = jnp.array(1.618034)
        y1, y2, f1, f2 = state.y1, state.y2, state.f1, state.f2
        f, e = state.f_info.f, state.jarrat_quotient


        p = (y1-y)**2*(f-f2)+(y2-y)**2*(f1-f)
        q = (y1-y)*(f-f2) + (y2-y)*(f1-f)
        e_eval = p/(q + 1e-15)
        y_jarrat = y + 0.5*e

        out_of_bounds = (y_jarrat < y1) | (y_jarrat > y2) 
        steps_getting_smaller = (jnp.abs(e_eval) < jnp.abs(e))
        e_big_enough = (1e-12 < jnp.abs(e)) # i dont see the point of this condition
        jarratt_not_usable = out_of_bounds | (1-steps_getting_smaller) | (1-e_big_enough)

        y_gss = y2 + (y1-y2)/phi
        #y_eval = jarratt_not_usable*y_gss + (1-jarratt_not_usable)*y_jarrat
        y_eval = jnp.where(jarratt_not_usable, y_gss, y_jarrat)
        f_eval, aux_eval = fn(y_eval, args)


        f_eval_bigger = (f_eval > f)
        y_eval_smaller = (y_eval < y)

        case1 = ((1-f_eval_bigger) & (1-y_eval_smaller))
        case2 = ((1-f_eval_bigger) & y_eval_smaller)
        case3 = (f_eval_bigger & y_eval_smaller)
        case4 = (f_eval_bigger & (1-y_eval_smaller))

        arr1 = jnp.array([y, y2, f, f2])
        arr2 = jnp.array([y1, y, f1, f])
        arr3 = jnp.array([y_eval, y2, f_eval, f2])
        arr4 = jnp.array([y1, y_eval, f1, f_eval])

        y1, y2, f1, f2 = case1*arr1 + case2*arr2 + case3*arr3 + case4*arr4
        
        case12 = (1-f_eval_bigger)
        arr12 = jnp.array([y_eval, f_eval])
        arr34 = jnp.array([y, f])
        
        y_eval, f_eval = case12*arr12 + (1-case12)*arr34
        f_eval_info = FunctionInfo.Eval(f_eval)


        def accepted():
            y_diff = (y_eval**ω - y**ω).ω
            f_diff = (f_eval**ω - state.f_info.f**ω).ω
            terminate = cauchy_termination(
                self.rtol, self.atol, self.norm, y_eval, y_diff, f_eval, f_diff
            )
            terminate = jnp.where(
                state.first_step, jnp.array(False), terminate
            )  # Skip termination on first step
            return y_eval, f_eval_info, aux_eval, terminate

        y, f_info, aux, terminate = accepted()
        result = RESULTS.successful

        state = _BrentState(
            y1=y1,
            y2=y2,
            f1=f1,
            f2=f2,
            jarrat_quotient=e_eval, #maybe only use e_eval when jarratt is used?
            f_info=f_info,
            first_step=jnp.array(False),
            terminate=terminate,
            result=result,
        )
        return y, state, aux
    


    def terminate(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _BrentState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        return state.terminate, state.result

    def postprocess(
        self,
        fn: Fn[Y, Scalar, Aux],
        y: Y,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _BrentState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Y, Aux, dict[str, Any]]:
        return y, aux, {}
    












