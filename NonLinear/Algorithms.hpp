/*==============================================================================
Algorithms

The various algorithms implemented by the NLopt library [1] can broadly be
divided into global or local optimisation. Furthermore there are algorithms
that do not need the gradient of the objective function or the constraints,
whereas other algorithms does require the gradients. Finally, many of the
algorithms come in different variants.

All various algorithms are identified by a flat enumerated list in the NLopt
library. This makes it difficult to understand the relation between a main
algorithm and its variants, and also the hierarchy of algorithms implemented.
Furthermore, it makes it impossible for the compiler to prevent illegal
combinations, and ensure that all the required components are defined. A
proper C++ interface should enable the compiler to enforce the correct
definitions expected by the NLopt library minimizing the risk of errors and
increasing code readability.

For the reason of readability, it is not possible to only state an algorithm
if this algorithm has variants. Then one must explicitly state that the
'Standard' variant of the algorithm is to be used. As an example, there are
multiple variants of the DIRECT algorithm, which is also implemented as is.
Specifying only 'Algorithm::Global::DIRECT' will not compile as one must say
'Algorithm::Global::DIRECT::Standard'. However, for some algorithms there is
no 'Standard' and one of the variants must be explicitly chosen.

The descriptions of the algorithms in the various header files are largely 
copied from the excellent NLopt documentatation.

References:

[1] Steven G. Johnson: The NLopt nonlinear-optimization package,
    http://ab-initio.mit.edu/nlopt

Author and Copyright: Geir Horn, 2018
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_NON_LINEAR_ALGORITHMS
#define OPTIMIZATION_NON_LINEAR_ALGORITHMS

#include <nlopt.h>

namespace Optimization::NonLinear
{

class Algorithm
{
public:

  // The algorithms are defined in the NLopt header as a simple enum, which 
  // means that any integer can be implicitly passed to any function taking 
  // the algorithm as argument, even values that are not corresponding to any 
  // algorithm. C++ introduces a scoped enumerator, making the enum a Type 
  // and therefore preventing unintended enum assignments. Hence the algorithm 
  // ID is redefined as a scoped enumerator. Since C++17 one has list 
  // initialisers for scoped enums having a storage type. Thus, one may create
  // an algorithm ID on the fly by saying Algorithm::ID{ 101 } even though 
  // there may not be an algorithm with number 101, it is still valid. 
  // There is no way to prevent such wilful out-of-range conversions 
  // and assignments. Defining the enum as a scoped type still helps though 
  // as it will prevent the assignment of negative numbers. The max number 
  // of algorithms is also explicitly defined so that the function creating 
  // the solver can test against this and throw an exception to prevent wilful 
  // wrong assignments.
  
  enum class ID : unsigned short int { MaxNumber = NLOPT_NUM_ALGORITHMS };
  
	// There is a function to test if a given algorithm requires gradients. In
	// the newer versions of the standard it is no longer necessary to initialise
	// variables of constant expressions, but some compilers still require this
	// initialisation.

	static constexpr bool Requires_Gradient( const ID TheAlogorithm )
	{
		bool Result = false;

		switch( static_cast< nlopt_algorithm >( TheAlogorithm ) )
	  {
			case NLOPT_GD_STOGO:
			case NLOPT_GD_STOGO_RAND:
			case NLOPT_LD_LBFGS_NOCEDAL:
			case NLOPT_LD_LBFGS:
			case NLOPT_LD_VAR1:
			case NLOPT_LD_VAR2:
			case NLOPT_LD_TNEWTON:
			case NLOPT_LD_TNEWTON_RESTART:
			case NLOPT_LD_TNEWTON_PRECOND:
			case NLOPT_LD_TNEWTON_PRECOND_RESTART:
			case NLOPT_GD_MLSL:
			case NLOPT_GD_MLSL_LDS:
			case NLOPT_LD_MMA:
			case NLOPT_LD_AUGLAG:
			case NLOPT_LD_AUGLAG_EQ:
			case NLOPT_LD_SLSQP:
			case NLOPT_LD_CCSAQ:
				Result = true;
				break;
			default:
				Result = false;
				break;
		}

		return Result;
	}

	// ---------------------------------------------------------------------------
	// Global Algorithms
	// ---------------------------------------------------------------------------
	//
	// The fundamental classification used here is whether the algorithm is global
	// or local. Whether it needs the gradients or not is encoded in the optimizer
	// classes towards the end of this header.

	struct Global
	{
		// DIviding RECTangles (DIRECT)
		// The algorithm is based on a systematic division of the search domain
		// into smaller and smaller hyper-rectangles. Besides the standard NLopt
		// implementation there is also a variant that does not assume equal weight
		// to all variable domains (problem dimensions) and the unscaled variant
		// could be better if there are large variations in the variable scales.
		// Finally, there implementation provided by the original proposers of the
		// algorithm can be chosen.
		//
		// There is also a family of DIRECT algorithms that are more biased towards
		// local search and could be faster for objective functions without too many
		// local minima. For this there are also Randomized versions, which uses
		// randomization to decide which dimension to halve when there are multiple
		// candidates of about the same weight.
    // 
    // The DIRECT variants are implemented in the DIRECT header

		struct DIRECT
		{
			static constexpr ID
				Standard = ID{ NLOPT_GN_DIRECT },
				Unscaled = ID{ NLOPT_GN_DIRECT_NOSCAL },
				Original = ID{ NLOPT_GN_ORIG_DIRECT };

			struct Local
			{
				static constexpr ID
					Standard   = ID{ NLOPT_GN_DIRECT_L },
					Randomized = ID{ NLOPT_GN_DIRECT_L_RAND },
					Original   = ID{ NLOPT_GN_ORIG_DIRECT_L };

				struct Unscaled
				{
					static constexpr ID
					  Standard   = ID{ NLOPT_GN_DIRECT_L_NOSCAL },
					  Randomized = ID{ NLOPT_GN_DIRECT_L_RAND_NOSCAL };
				};
			};
		};

		// Controlled Random Search
		// The CRS algorithms are sometimes compared to genetic algorithms, in that
		// they start with a random "population" of points, and randomly "evolve"
		// these points by heuristic rules. In this case, the "evolution" somewhat
		// resembles a randomized Nelder-Mead algorithm. There are no variants of
		// this algorithm, and so it is directly defined.
    // 
    // This is implemented in the header with the same name

		static constexpr ID ControlledRandomSearch = ID{ NLOPT_GN_CRS2_LM };

		// Multi-Level Single-Linkage
		// MLSL is a "multistart" algorithm: it works by doing a sequence of local
		// optimizations (using some other local optimization algorithm) from
		// random starting points. The low-discrepancy sequence (LDS) can be used
		// instead of pseudo random numbers, which arguably improves the convergence
		// rate
    // 
    // These variants are provided in the header with the same name

		struct MultiLevelSingleLinkage
		{
			static constexpr ID
				NonDerivative = ID{ NLOPT_GN_MLSL },
				Derivative    = ID{ NLOPT_GD_MLSL };

			struct LowDiscrepancySequence
			{
				static constexpr ID
					NonDerivative = ID{ NLOPT_GN_MLSL_LDS },
					Derivative    = ID{ NLOPT_GD_MLSL_LDS };
			};
		};

		// Stochastic Global Optimiser
		// The StoGo uses a technique similar to the multi-level single linkage by
		// dividing the search space into hyper rectangles by a branch-and-bound
		// technique and then search each of them with a gradient based local
		// algorithm. The random variant uses "some randomness" in the search.
    // 
    // The specialisations are implemented in the StoGo header file

		struct StoGo
		{
			static constexpr ID
			  Standard   = ID{ NLOPT_GD_STOGO },
				Randomized = ID{ NLOPT_GD_STOGO_RAND };
		};

	}; // Structure for global algorithms.

	// ---------------------------------------------------------------------------
	// Local Algorithms
	// ---------------------------------------------------------------------------
	//
  // Local algorithms do not guarantee to find global optima, but depending on 
  // the objective function the local search may convert to a global optimum.
  // The provided algorithms may broadly be classified as requiring a gradient 
  // or not requiring the gradient.
  
  struct Local
  {
    // The quadratic approximation algorithms are due to M. J. D. Powell and 
    // is based on approximating the objective functions by either linear or 
    // quadratic functions. All variants support bound constraints, and the 
    // linear approximation also supports inequality and equality constraints.
    // The quadratic variants may perform poorly for functions that are not 
    // twice differentiable. These algorithms are implemented in the header 
    // file Local Approximations.
    
    struct NonDerivative
    {
      static constexpr ID
        LinearApproximation    = ID{ NLOPT_LN_COBYLA },
        QuadraticApproximation = ID{ NLOPT_LN_NEWUOA_BOUND },
        RescalingApproximation = ID{ NLOPT_LN_BOBYQA };
    };
  };

};

/*==============================================================================

 Solver pointer

==============================================================================*/
//
// The Optimiser has an NLopt optimiser object. However, this object cannot
// be initialised from the constructor as the dimensionality of the problem
// is generally not known, and the virtual functions for defining the problem
// like the objective function and constraint functions are generally not
// defined at the time the optimizer's constructor executes. The solution is
// therefore to dynamically allocate the solver once on first usage. See
// the details in the Create Solver function in the Optimizer header.

using SolverPointer = nlopt_opt;

}       // end name space Optimization non-linear
#endif  // OPTIMIZATION_NON_LINEAR_ALGORITHMS
