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
just copied from the excellent NLopt documentatation, and the copyright of
the descriptions belongs to Steven G. Johnson.

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
private:

  // The algorithms are defined in the NLopt header as a simple enum, which
  // means that any integer can be implicitly passed to any function taking
  // the algorithm as argument, even values that are not corresponding to any
  // algorithm. C++ introduces a scoped enumerator, making the enum a Type
  // and therefore preventing unintended enum assignments. Hence the algorithm
  // identifier is redefined as a scoped enumerator. Since C++17 one has list
  // initialisers for scoped enums having a storage type. Thus, one may create
  // an algorithm identifier on the fly by saying Algorithm::Identifier{ 101 }
	// even though there may not be an algorithm with number 101, it is still
	// valid. There is no way to prevent such wilful out-of-range conversions
  // and assignments. Defining the enum as a scoped type still helps though
  // as it will prevent the assignment of negative numbers. The max number
  // of algorithms is also explicitly defined so that the function creating
  // the solver can test against this and throw an exception to prevent wilful
  // wrong assignments.
  //
  // It should be noted that although all the algorithms below are defined
  // structurally to highlight what they do and not their names, one may still
  // use the NLopt defined constants by encapsulating them in the identifier
	// enumeration as is done in the actual definitions below. By default it
	// defines the number of algorithms and one special field for no algorithm
	// in case the optimiser has no secondary algorithm.

  enum class Identifier : unsigned short int {
		MaxNumber = NLOPT_NUM_ALGORITHMS,
		NoAlgorithm
	};

public:

	// The Identifier class is made private to ensure that the wrong extensions
	// will not be made, as indicated. However, that also means that it cannot
	// be used as arguments to functions or templates. The following is a small
	// trick that will allow Algorithm::ID to be used as a legal type, but all
	// values have to be assigned from the set of constants declared
	// hierarchically below to highlight their structural dependencies.

	using ID = Identifier;

	// There is a function to test if a given algorithm requires gradients. In
	// the newer versions of the standard it is no longer necessary to initialise
	// variables of constant expressions, but some compilers still require this
	// initialisation.

	static constexpr bool RequiresGradient( const ID TheAlogorithm )
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

	//  The the derivative free algorithms can be identified simply by negating the
	//  above function.

	static constexpr bool NoGradient( const ID TheAlogorithm )
	{ return !RequiresGradient( TheAlogorithm ); }

	// Another view of the algorithms is in terms of their support for
  // constraints: Some algorithms supports only inequality constraints, some
  // supports bound constraints, some supports equality constraints. The
	// following functions allows to check the kind of constraints supported by
	// an algorithm.

	static constexpr bool SupportsInequalityConstraints( const ID TheAlogorithm )
	{
		bool Result = false;

		switch( static_cast< nlopt_algorithm >( TheAlogorithm ) )
	  {
		  case NLOPT_GN_ORIG_DIRECT:
		  case NLOPT_GN_ISRES:
			case NLOPT_LN_COBYLA:
		  case NLOPT_LD_MMA:
			case NLOPT_LD_SLSQP:
			  Result = true;
				break;
		  default:
			  Result = false;
				break;
	  };

		return Result;
	}

	static constexpr bool SupportsEqualityConstraints( const ID TheAlogorithm )
	{
		bool Result = false;

		switch( static_cast< nlopt_algorithm >( TheAlogorithm ) )
	  {
		  case NLOPT_GN_ISRES:
			case NLOPT_LN_COBYLA:
		  case NLOPT_LD_SLSQP:
			  Result = true;
			break;
		  default:
			  Result = false;
			break;
	  };

		return Result;
	}

  static constexpr bool SupportsBoundConstraints( const ID TheAlogorithm )
	{
		bool Result = false;

		switch( static_cast< nlopt_algorithm >( TheAlogorithm ) )
	  {
		  case NLOPT_GN_DIRECT:
		  case NLOPT_GN_DIRECT_NOSCAL:
		  case NLOPT_GN_ORIG_DIRECT:
		  case NLOPT_GN_DIRECT_L:
		  case NLOPT_GN_DIRECT_L_RAND:
		  case NLOPT_GN_ORIG_DIRECT_L:
		  case NLOPT_GN_DIRECT_L_NOSCAL:
		  case NLOPT_GN_DIRECT_L_RAND_NOSCAL:
		  case NLOPT_GN_CRS2_LM:
			case NLOPT_GN_MLSL:
			case NLOPT_GD_MLSL:
			case NLOPT_GN_MLSL_LDS:
			case NLOPT_GD_MLSL_LDS:
			case NLOPT_GD_STOGO:
		  case NLOPT_GD_STOGO_RAND:
			case NLOPT_GN_ESCH:
		  case NLOPT_LN_COBYLA:
		  case NLOPT_LN_BOBYQA:
			case NLOPT_LN_NEWUOA_BOUND:
		  case NLOPT_LN_NELDERMEAD:
			case NLOPT_LN_SBPLX:
		  case NLOPT_LD_SLSQP:
				Result = true;
			  break;
		  default:
			  Result = false;
				break;
	  };

		return Result;
	}

	// Unconstrained algorithms do not support any constraint by definition and
	// the test function is therefore defined in terms of the other constraint
	// functions

	static constexpr bool Unconstrained( const ID TheAlogorithm )
	{
		return !( SupportsInequalityConstraints( TheAlogorithm ) ||
	            SupportsEqualityConstraints  ( TheAlogorithm ) ||
	            SupportsBoundConstraints     ( TheAlogorithm ) );
	}

	//  Finally there are some algorithms that requires a subsidiary algorithm
	//  to solve the problem. The following function tests if this is the case

	static constexpr bool RequiresSubsidiary( const ID TheAlogorithm )
	{
		bool Result = false;

		switch( static_cast< nlopt_algorithm >( TheAlogorithm ) )
		{
			case NLOPT_GN_MLSL:
		  case NLOPT_GD_MLSL:
			case NLOPT_GN_MLSL_LDS:
		  case NLOPT_GD_MLSL_LDS:
			case NLOPT_AUGLAG:
		  case NLOPT_AUGLAG_EQ:
			  Result = true;
				break;
			default:
		    Result = false;
			  break;
		};

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

    // AGS
    // The algorithm can handle arbitrary objectives and non-linear inequality
    // constraints. Also bound constraints are required for this method. To
    // guarantee convergence, objectives and constraints should satisfy the
    // Lipschitz condition on the specified hyper-rectangle. This means that the
    // function should have bounded first derivatives. Note: The AGS algorithm
    // can only be used for low dimensional problems with less than
    // 6 dimensions, but supports both bound constraints and individual
    // inequality constraints.
    //
    // The specialisation is implemented in the AGS header file. However,
    // the constant identifying this algorithm is not defined, and hence
    // it cannot be used. It can be included from the original implementation,
    // see the header file.

    // static constexpr ID AGS = ID{ NLOPT_GN_AGS };

    // Evolutionary Algorithm
    // The algorithm uses and evolutionary strategy to implement a global
    // optimization for bound constrained problems. The specialisation is
    // implemented in the Evolutionary header file.

    static constexpr ID Evolutionary = ID{ NLOPT_GN_ESCH };

    // Finally it is possible to combine the above algorithms with the augmented
    // Lagrangian method adding a penalty to the objective function whenever the
    // point is outside of the feasible region and the uses a secondary
		// algorithm to solve the unconstrained problem. There are two variants,
		// one adding the penalty for both the inequality and the equality
	  // constraints, and one that adds the penalty only for the equality
	  // constraints and passes the inequality constraints to the secondary
	  // algorithm, which must then support inequality constraints.

	  struct Penalty
	  {
			 static constexpr ID
	       AllConstraints      = ID{ NLOPT_AUGLAG },
	       EqualityConstraints = ID{ NLOPT_AUGLAG_EQ };
		};

	}; // Structure for global algorithms.

  // There is a utility function to test if an algorithm is global or not
  // following the template of the above functions to check aspects of the
  // algorithms.

  static constexpr bool IsGlobal( const ID TheAlogorithm )
  {
		bool Result = false;

		switch( static_cast< nlopt_algorithm >( TheAlogorithm ) )
		{
		  case NLOPT_GN_DIRECT:
		  case NLOPT_GN_DIRECT_NOSCAL:
		  case NLOPT_GN_ORIG_DIRECT:
		  case NLOPT_GN_DIRECT_L:
		  case NLOPT_GN_DIRECT_L_RAND:
		  case NLOPT_GN_ORIG_DIRECT_L:
		  case NLOPT_GN_DIRECT_L_NOSCAL:
		  case NLOPT_GN_DIRECT_L_RAND_NOSCAL:
		  case NLOPT_GN_CRS2_LM:
			case NLOPT_GN_MLSL:
			case NLOPT_GD_MLSL:
			case NLOPT_GN_MLSL_LDS:
			case NLOPT_GD_MLSL_LDS:
			case NLOPT_GD_STOGO:
		  case NLOPT_GD_STOGO_RAND:
			case NLOPT_GN_ESCH:
		  case NLOPT_AUGLAG:
			case NLOPT_AUGLAG_EQ:
		    Result = true;
	    default:
			  Result = false;
		};

		return Result;
	}

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
    // Three approximation algorithms are due to M. J. D. Powell and
    // is based on approximating the objective functions by either linear or
    // quadratic functions. All variants support bound constraints, and the
    // linear approximation also supports inequality and equality constraints.
    // The quadratic variants may perform poorly for functions that are not
    // twice differentiable.

    struct Approximation
    {
      static constexpr ID
        Linear    = ID{ NLOPT_LN_COBYLA },
        Quadratic = ID{ NLOPT_LN_NEWUOA_BOUND },
        Rescaling = ID{ NLOPT_LN_BOBYQA };
    };

    // Evolutionary stochastic ranking
    // The method supports bound constraints and both equality and inequality
    // constraints. It is claimed to be a global optimisation algorithm, but
    // as there is no proof of convergence it is classified as a local
    // algorithm in this interface library. The specialisation is implemented
    // in the Evolutionary header file.

    static constexpr ID Evolutionary = ID{ NLOPT_GN_ISRES };

    // The principal axis method is fundamentally for unconstrained problems.
    // Steven G. Johnson implements support for bound constraints by returning
    // infinity for the object function if the argument is outside of the
    // bounded region. Naturally, this leads to slow convergence, and bound
    // constraints are therefore not supported by the implemented specialisation
    // for this algorithm that is provided in the Principal Axis header.
    // Johnson recommends using either the liner or rescaling approximation
    // algorithms for bounded problems.

    static constexpr ID PrincipalAxis = ID{ NLOPT_LN_PRAXIS };

    // The classical simplex algorithm by Nelder and Mead is also used on a
    // sequence of sub-spaces by the Subplex algorithm which is implemented in
    // NLOpt under a different name. The interface specialisations are available
    // in the Simplex header. Both variants only supports bound constraints.

    struct Simplex
    {
      static constexpr ID
        NelderMead = ID{ NLOPT_LN_NELDERMEAD },
        Subspace   = ID{ NLOPT_LN_SBPLX };
    };

    // Quasi-Newton algorithms are variants of the Broyden–Fletcher–Goldfarb–
		// Shanno (BFGS) algorithm that is a hill-climbing optimization techniques
	  // that seek a stationary point of the objective function where the gradient
	  // is zero. It solves the Newton equation to find the search direction,
	  // however by using an approximation of the Hessian matrix. There are three
	  // variants implemented: One aiming to minimize the use of memory to store
    // the approximations to the Hessian matrix, a variable metric algorithm,
		// and several algorithms using a truncated version of Newton's algorithm.
	  // All of these are for unconstrained problems.
	  //
	  // Finally, there is a quadratic programming method supporting both
	  // inequality and equality constraints. This uses a dense matrix approach
	  // to store the approximation of the Hessian matrix, and will therefore
	  // require more memory.
	  //
	  // All variants are implemented in the Quasi-Newton header.

    struct QuasiNewton
    {
			static constexpr ID
			  LowMemeory           = ID{ NLOPT_LD_LBFGS },
			  QuadraticProgramming = ID{ NLOPT_LD_SLSQP };

		  struct VariableMetric
		  {
				static constexpr ID
				  RankOne = ID{ NLOPT_LD_VAR1 },
				  RankTwo = ID{ NLOPT_LD_VAR2 };
			};

			struct Truncated
			{
				static constexpr ID
				  PreconditionRestart = ID{ NLOPT_LD_TNEWTON_PRECOND_RESTART },
				  Precondition        = ID{ NLOPT_LD_TNEWTON_PRECOND },
				  Restart             = ID{ NLOPT_LD_TNEWTON_RESTART },
				  Plain               = ID{ NLOPT_LD_TNEWTON };
			};
		};

		// The most general way to include constraints is to add a penalty to the
		// objective function when it moves outside of the feasible region. Note
		// that this requires the function to exist outside the feasible region.
		// The most general of these approaches is known as the Augmented Lagrangian
		// method that uses any of the above unconstrained solvers to optimize an
		// objective function consisting of the original objective function
	  // augmented with the penalty for the constraints. If the solution obtained
    // by the sub-solver gives an infeasible point, i.e. a point violating the
    // constraints, then the size of the penalty is increased and the augmented
    // problem is solved again. There are two variants of the augmented
		// Lagrangian method: one that sets a penalty for both the inequality and
		// the equality constraints, and one that sets the penalty only for the
		// equality constraints and passes the inequality constraints through to
		// the sub-solver that must support inequality constraints in this case.
		//
		// The convex approximation was proposed by K. Svanberg and uses a
	  // quadratic penalty term requiring the gradient of the objective
    // function. This algorithm has later been improved, and the improved
		// version is called the method of the moving asymptotes. Both algorithms
	  // support inequality constraints, but it is not clear if they support
    // bound constraints. However support for bound constraints is provided in
		// the interface specialisation for both methods.
		//
		// The penalty specialisations are implemented in the Penalty header

		struct Penalty
		{
       static constexpr ID
	       AllConstraints      = ID{ NLOPT_AUGLAG },
	       EqualityConstraints = ID{ NLOPT_AUGLAG_EQ },
	       ConvexSeparable     = ID{ NLOPT_LD_CCSAQ },
	       MovingAsymptotes    = ID{ NLOPT_LD_MMA };
		};
  };

  // The function testing if an algorithm is local simply negates the test for
  // the global algorithm. However, the augmented Lagrangian is both a global
  // or a local method depending on the secondary optimisation algorithm.

  static constexpr bool IsLocal( const ID TheAlogorithm )
  {
		bool Result = false;

		switch( static_cast< nlopt_algorithm >( TheAlogorithm ) )
		{
		  case NLOPT_AUGLAG:
			case NLOPT_AUGLAG_EQ:
		    Result = true;
	    default:
			  Result = !IsGlobal( TheAlogorithm );
		};

		return Result;
	}

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
