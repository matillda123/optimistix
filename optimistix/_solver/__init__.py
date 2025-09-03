from .backtracking import BacktrackingArmijo as BacktrackingArmijo
from .best_so_far import (
    BestSoFarFixedPoint as BestSoFarFixedPoint,
    BestSoFarLeastSquares as BestSoFarLeastSquares,
    BestSoFarMinimiser as BestSoFarMinimiser,
    BestSoFarRootFinder as BestSoFarRootFinder,
)
from .bisection import Bisection as Bisection
from .broyden import BadBroyden as BadBroyden, Broyden as Broyden
from .brent import (
    Brent as Brent,
    GoldenSectionSearch as GoldenSectionSearch, 
    Jarratt as Jarratt
)
from .brent_dekker import (
    BrentDekker as BrentDekker,
    InverseQuadraticInterpolation as InverseQuadraticInterpolation, 
    Secant1D as Secant1D
)
from .dogleg import Dogleg as Dogleg, DoglegDescent as DoglegDescent
from .fixed_point import FixedPointIteration as FixedPointIteration
from .gauss_newton import (
    AbstractGaussNewton as AbstractGaussNewton,
    GaussNewton as GaussNewton,
    NewtonDescent as NewtonDescent,
)
from .gradient_methods import (
    AbstractGradientDescent as AbstractGradientDescent,
    GradientDescent as GradientDescent,
    ScaledGradientDescent as ScaledGradientDescent,
    SteepestDescent as SteepestDescent,
)
from .learning_rate import (
    LearningRate as LearningRate,
    pade_10 as pade_10,
    pade_20 as pade_20,
    pade_11 as pade_11,
    pade_01 as pade_01,
    pade_02 as pade_02,
    ScaledLearningRate as ScaledLearningRate,
)
from .levenberg_marquardt import (
    DampedNewtonDescent as DampedNewtonDescent,
    IndirectDampedNewtonDescent as IndirectDampedNewtonDescent,
    IndirectLevenbergMarquardt as IndirectLevenbergMarquardt,
    LevenbergMarquardt as LevenbergMarquardt,
)
from .limited_memory_bfgs import AbstractLBFGS as AbstractLBFGS, LBFGS as LBFGS
from .nelder_mead import NelderMead as NelderMead
from .newton_chord import Chord as Chord, Newton as Newton
from .nonlinear_cg import (
    dai_yuan as dai_yuan,
    fletcher_reeves as fletcher_reeves,
    hestenes_stiefel as hestenes_stiefel,
    NonlinearCG as NonlinearCG,
    NonlinearCGDescent as NonlinearCGDescent,
    polak_ribiere as polak_ribiere,
)
from .optax import OptaxMinimiser as OptaxMinimiser
from .quasi_newton import (
    AbstractBFGS as AbstractBFGS,
    AbstractDFP as AbstractDFP,
    AbstractQuasiNewton as AbstractQuasiNewton,
    BFGS as BFGS,
    DFP as DFP,
)
from .secant import Secant as Secant, FalsePosition as FalsePosition
from .trust_region import (
    ClassicalTrustRegion as ClassicalTrustRegion,
    LinearTrustRegion as LinearTrustRegion,
)
