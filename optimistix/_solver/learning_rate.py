from typing import cast
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Scalar, ScalarLike

from equinox.internal import ω

from .._custom_types import Y
from .._search import AbstractSearch, FunctionInfo
from .._solution import RESULTS


def _typed_asarray(x: ScalarLike) -> Array:
    return jnp.asarray(x)


class LearningRate(AbstractSearch[Y, FunctionInfo, FunctionInfo, None]):
    """Move downhill by taking a step of the fixed size `learning_rate`."""

    learning_rate: ScalarLike = eqx.field(converter=_typed_asarray)

    def init(self, y: Y, f_info_struct: FunctionInfo) -> None:
        return None

    def step(
        self,
        first_step: Bool[Array, ""],
        y: Y,
        y_eval: Y,
        f_info: FunctionInfo,
        f_eval_info: FunctionInfo,
        state: None,
    ) -> tuple[Scalar, Bool[Array, ""], RESULTS, None]:
        del first_step, y, y_eval, f_info, f_eval_info, state
        learning_rate = cast(Array, self.learning_rate)
        return learning_rate, jnp.array(True), RESULTS.successful, None


LearningRate.__init__.__doc__ = """**Arguments:**

- `learning_rate`: The fixed step-size used at each step.
"""

















# same as 1. order taylor
def pade_10(L: float, L_prime: float, grad_dot_pk: float) -> float:
    return (L_prime - L)/(2*grad_dot_pk)


def pade_01(L: float, L_prime: float, grad_dot_pk: float) -> float:
    return L/L_prime * (L_prime - L)/(2*grad_dot_pk)



# pade_20, pade_11 and pade_02 assume a (quasi)-newton like step in their derivation

# same as 2. order taylor
def pade_20(L: float, L_prime: float, grad_dot_pk: float) -> float:
    diskriminante = 1 - (L_prime - L)/(2*grad_dot_pk)
    return 2*(1 - jnp.sign(diskriminante)*jnp.sqrt(jnp.abs(diskriminante)))


def pade_11(L: float, L_prime: float, grad_dot_pk: float) -> float:
    return 2*(L_prime - L)/(4*grad_dot_pk - (L_prime - L))

def pade_02(L: float, L_prime: float, grad_dot_pk: float) -> float:
    diskriminante = 1 - 4*(1 + L/(4*grad_dot_pk)) * (L_prime - L)/L_prime
    return L/(4*grad_dot_pk + L) * (1 + jnp.sign(diskriminante)*jnp.sqrt(jnp.abs(diskriminante)))



class ScaledLearningRate(AbstractSearch[Y, FunctionInfo, FunctionInfo, None]):
    """Move downhill by taking a step of `learning_rate` which is scaled based on a Taylor/Pade-Approximation of the Loss-function. 
    The type of approximation is controlled via `func_approx`."""

    learning_rate: ScalarLike = eqx.field(converter=_typed_asarray)
    scaling_rate: ScalarLike = eqx.field(converter=_typed_asarray)
    lower_bound: ScalarLike = eqx.field(converter=_typed_asarray)
    func_approx: Callable[[Scalar, Scalar, Scalar], Scalar]

    def init(self, y: Y, f_info_struct: FunctionInfo) -> None:
        return None

    def step(
        self,
        first_step: Bool[Array, ""],
        y: Y,
        y_eval: Y,
        f_info: FunctionInfo,
        f_eval_info: FunctionInfo,
        state: None,
    ) -> tuple[Scalar, Bool[Array, ""], RESULTS, None]:
        
        if isinstance(f_eval_info, FunctionInfo.Eval):
            raise ValueError(
                "Cannot use `ScaledLearningRate` with this solver. This is because "
                "`ScaledLearningRate` requires gradients of the target function, but "
                "this solver does not evaluate such gradients."
            )

        elif isinstance(f_eval_info, FunctionInfo.EvalGrad):
            grad_dot_pk = f_eval_info.compute_grad_dot(-1*f_eval_info.grad)
            
        else:
            y_diff = (y_eval**ω - y**ω).ω
            grad_dot_pk = f_eval_info.compute_grad_dot(y_diff)


        L = f_eval_info.as_min()
        #L_prime = -1*L*self.scaling_rate # approximate lower bound for the minimal value, is problematic when minimum is < 0
        L_prime = self.lower_bound

        # this "+ scaling_rate * jnp.abs(L_prime - L)" is done heuristically to avoid divergences if grad_dot_pk -> 0
        grad_dot_pk = grad_dot_pk + self.scaling_rate*jnp.abs(L_prime - L)
        eta = self.func_approx(L, L_prime, grad_dot_pk)

        step_size = self.learning_rate * jnp.abs(eta)
        step_size = cast(Array, step_size)
        return step_size, jnp.array(True), RESULTS.successful, None



ScaledLearningRate.__init__.__doc__ = """**Arguments:**

- `learning_rate`: The fixed step-size used at each step.
- `func_approx`: A callable that computes the appropriate step scaling based on the Pade-approximation of the loss-function. 
"""
