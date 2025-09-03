import functools as ft
from collections.abc import Callable
from typing import Any, ClassVar, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PyTree, Scalar

from .._custom_types import Aux, Fn
from .._root_find import AbstractRootFinder
from .._solution import RESULTS





class _IQIState(eqx.Module):
    y1: Scalar
    y2: Scalar
    val1: Float[Array, ""]
    val2: Float[Array, ""]



def _do_iqi(y1: float, y2: float, y3: float, f1: float, f2: float, f3: float) -> float:
    # adding 1e-15 here to avoid python raising errors
    return y3 * f2*f1/((f3-f2)*(f3-f1) + 1e-15) + y2 * f3*f1/((f2-f3)*(f2-f1) + 1e-15) + y1 * f3*f2/((f1-f3)*(f1-f2) + 1e-15)



class InverseQuadraticInterpolation(AbstractRootFinder[Scalar, Scalar, Aux, _IQIState]):
    """The inverse quadratic interpolation method of root finding. This may only be used with functions
    `R->R`, i.e. functions with scalar input and scalar output.

    This requires the following `options`:

    - `y1`: An additional initial guess value.
    - `y2`: Another additional initial guess value.

    Which are passed as, for example,
    `optimistix.root_find(..., options=dict(y1=0, y2=1))`

    An inverse quadratic polynomial is constructed based on the three points `[y1, y2]` and `y` as well as their corresponding function values. 
    The analytically found root of this function is an approximate of a root of the target function. It is used in the next iteration. 
    The inverse quadratic function only ever possesses one root which is always real. The method is unstable due to the possibility of division by zero. 
    The iteration stops once the interval of `[y, y1]` and the function-value are sufficiently small.

    The method can be made stable by introducing a conditional alternative update. This is exactly what Brent's method and others are doing.
    """

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
    ) -> _IQIState:
        y1 = jnp.asarray(options["y1"], f_struct.dtype)
        y2 = jnp.asarray(options["y2"], f_struct.dtype)
        del options, aux_struct
        if jnp.shape(y) != () or jnp.shape(y1) != () or jnp.shape(y2) != ():
            raise ValueError(
                "InverseQuadraticInterpolation can only be used to find the roots of a function taking a "
                "scalar input."
            )
        if not isinstance(f_struct, jax.ShapeDtypeStruct) or f_struct.shape != ():
            raise ValueError(
                "InverseQuadraticInterpolation can only be used to find the roots of a function producing "
                "a scalar output."
            )
        
        val1, aux = fn(y1, args)
        val2, aux = fn(y2, args)

        return _IQIState(
            y1=y1,
            y2=y2,
            val1=val1,
            val2=val2,
        )

    def step(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: _IQIState,
        tags: frozenset[object],
    ) -> tuple[Scalar, _IQIState, Aux]:
        del options

        val, aux = fn(y, args)
        new_y = _do_iqi(y, state.y1, state.y2, val, state.val1, state.val2)

        new_state = _IQIState(
            y1=y,
            y2=state.y1,
            val1=val,
            val2=state.val1,
        )
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: _IQIState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        del fn, args, options
        scale = self.atol + self.rtol * jnp.abs(y)
        y_small = jnp.abs(state.y1 - y) < scale
        f_small = jnp.abs(state.val1) < self.atol
        return y_small & f_small, RESULTS.successful

    def postprocess(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _IQIState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Scalar, Aux, dict[str, Any]]:
        return y, aux, {}





















class _Secant1DState(eqx.Module):
    y1: Scalar
    val: Float[Array, ""]



def _do_secant(y: float, y1: float, val: float, val1: float) -> float:
    # adding 1e-15 here to avoid python raising errors
    return y - val*(y1 - y)/(val1 - val + 1e-15)




class Secant1D(AbstractRootFinder[Scalar, Scalar, Aux, _Secant1DState]):
    """The classical secant method of root finding. This may only be used with functions
    `R->R`, i.e. functions with scalar input and scalar output.

    This requires the following `options`:

    - `y1`: An additional additional guess.

    Which are passed as, for example,
    `optimistix.root_find(..., options=dict(y1=0))`

    Using the root of a linear interpolation between the current and previous iterate an updated guess is found.
    The iteration stops once the change of `y` is sufficiently small. The method is unstable due to the possibility of division by zero.

    """

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
    ) -> _Secant1DState:
        y1 = jnp.asarray(options["y1"], f_struct.dtype)
        del options, aux_struct
        if jnp.shape(y) != () or jnp.shape(y1) != ():
            raise ValueError(
                "Secant method can only be used to find the roots of a function taking a "
                "scalar input."
            )
        if not isinstance(f_struct, jax.ShapeDtypeStruct) or f_struct.shape != ():
            raise ValueError(
                "Secant method can only be used to find the roots of a function producing "
                "a scalar output."
            )
        
        val, aux = fn(y1, args)

        return _Secant1DState(
            y1 = y1,
            val = val,
        )

    def step(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: _Secant1DState,
        tags: frozenset[object],
    ) -> tuple[Scalar, _Secant1DState, Aux]:
        del options, tags

        val, aux = fn(y, args)
        new_y = _do_secant(y, state.y1, val, state.val)
        new_state = _Secant1DState(
            y1 = y,
            val = val
        )
        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: _Secant1DState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        del fn, args, options
        scale = self.atol + self.rtol * jnp.abs(y)
        y_small = jnp.abs(state.y1 - y) < scale
        f_small = jnp.abs(state.val) < self.atol
        return y_small & f_small, RESULTS.successful

    def postprocess(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _Secant1DState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Scalar, Aux, dict[str, Any]]:
        return y, aux, {}




































class _BrentDekkerState(eqx.Module):
    y0: Scalar
    y1: Scalar
    y2: Scalar
    val: Float[Array, ""]
    val0: Float[Array, ""]
    val1: Float[Array, ""]
    use_bisection: Bool[Array, ""]




class BrentDekker(AbstractRootFinder[Scalar, Scalar, Aux, _BrentDekkerState]):
    """The Brent-Dekker Algorithm for root finding. Also known as Brent's method but not to be 
    confused with Brent's method for minimization. This may only be used with functions
    `R->R`, i.e. functions with scalar input and scalar output.

    This requires the following `options`:

    - `lower`: The lower bound on the interval which contains the root.
    - `upper`: The upper bound on the interval which contains the root.

    Which are passed as, for example,
    `optimistix.root_find(..., options=dict(lower=0, upper=1))`

    This algorithm combines Inverse Quadratic Interpolation (IQI), the Secant method and the Bisection method. The Bisection method provides guaranteed convergence, 
    while IQI and the Secant method introduce a superlinear convergence rate. Each iteration the algorithm will attempt an update based on either IQI or the 
    Secant method. If necessary the Bisection method is used to avoid divergence or stagnation. 
    The iteration stops once the change of `y` is sufficiently small.

    """

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
    ) -> _BrentDekkerState:
        
        lower = jnp.asarray(options["lower"], f_struct.dtype)
        upper = jnp.asarray(options["upper"], f_struct.dtype)
        del options, aux_struct
        if jnp.shape(y) != () or jnp.shape(lower) != () or jnp.shape(upper) != ():
            raise ValueError(
                "BrentDekker can only be used to find the roots of a function taking a "
                "scalar input."
            )
        if not isinstance(f_struct, jax.ShapeDtypeStruct) or f_struct.shape != ():
            raise ValueError(
                "BrentDekker can only be used to find the roots of a function producing "
                "a scalar output."
            )
            
        val, _ = fn(y, args)
        lower_val, _ = fn(lower, args)
        upper_val, _ = fn(upper, args)


        lower_neg = lower_val < 0
        upper_neg = upper_val < 0
        root_not_contained = lower_neg == upper_neg
        val = eqx.error_if(
            val,
            root_not_contained,
            msg="The root is not contained in [lower, upper]",
        )


        same_sign_lower = (jnp.sign(val)*jnp.sign(lower_val) > 0)

        y0 = same_sign_lower*upper + (1-same_sign_lower)*lower
        val0 = same_sign_lower*upper_val + (1-same_sign_lower)*lower_val

        y1 = y2 = y
        val1 = val
        
        return _BrentDekkerState(
            y0 = y0,
            y1 = y1,
            y2 = y2,
            val = val,
            val0 = val0,
            val1 = val1,
            use_bisection = jnp.array(True)
        )

    def step(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: _BrentDekkerState,
        tags: frozenset[object],
    ) -> tuple[Scalar, _BrentDekkerState, Aux]:
        del options, tags

        y0, y1, y2, val, val0, val1 = state.y0, state.y1, state.y2, state.val, state.val0, state.val1

        use_iqi = (val!=val1) & (val0!=val1)
        #new_y = use_iqi * _do_iqi(y, y0, y1, val, val0, val1) + (1-use_iqi) * _do_secant(y, y0, val, val0)
        new_y = jnp.where(use_iqi, 
                          _do_iqi(y, y0, y1, val, val0, val1), 
                          _do_secant(y, y0, val, val0))


        tol = self.atol + self.rtol * jnp.abs(y)
        min1 = jnp.abs(new_y - y)
        min2 = jnp.abs(y - y1)
        min3 = jnp.abs(y1 - y2)
        use_bisection = state.use_bisection

        cond1 = ((new_y < (3*y0 + y)/4) & (y < new_y))
        cond2 = (use_bisection & (min1 >= min2/2))
        cond3 = ((1 - use_bisection) & (min1 >= min3/2))
        cond4 = (use_bisection & (min2 < tol))
        cond5 = ((1 - use_bisection) & (min3 < tol))
        use_bisection = (cond1 | cond2 | cond3 | cond4 | cond5).astype(jnp.bool_)

        #new_y = use_bisection*(0.5 * (y0 + y)) + (1-use_bisection)*new_y
        new_y = jnp.where(use_bisection, 0.5 * (y0 + y), new_y)

        new_val, aux = fn(new_y, args)
        y1, y2, val1 = y, y1, val


        opposite_sign = (jnp.sign(val0)*jnp.sign(new_val) < 0)

        y = jnp.where(opposite_sign, new_y, y)
        val = jnp.where(opposite_sign, new_val, val)

        y0 = jnp.where(1-opposite_sign, new_y, y0)
        val0 = jnp.where(1-opposite_sign, new_val, val0)



        is_smaller = (jnp.abs(val0) < jnp.abs(val))

        y0_new = jnp.where(is_smaller, y0, y)
        new_y = jnp.where(is_smaller, y, y0)

        val0_new = jnp.where(is_smaller, val0, val)
        val_new = jnp.where(is_smaller, val, val0)

        new_state = _BrentDekkerState(
            y0 = y0_new,
            y1 = y1,
            y2 = y2,
            val = val_new,
            val0 = val0_new,
            val1 = val1,
            use_bisection = use_bisection
        )

        return new_y, new_state, aux

    def terminate(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        args: PyTree,
        options: dict[str, Any],
        state: _BrentDekkerState,
        tags: frozenset[object],
    ) -> tuple[Bool[Array, ""], RESULTS]:
        del fn, args, options
        scale = self.atol + self.rtol * jnp.abs(y)
        y_small = jnp.abs(state.y0 - y) < scale
        f_small = jnp.abs(state.val) < self.atol
        return y_small & f_small, RESULTS.successful

    def postprocess(
        self,
        fn: Fn[Scalar, Scalar, Aux],
        y: Scalar,
        aux: Aux,
        args: PyTree,
        options: dict[str, Any],
        state: _BrentDekkerState,
        tags: frozenset[object],
        result: RESULTS,
    ) -> tuple[Scalar, Aux, dict[str, Any]]:
        return y, aux, {}



















