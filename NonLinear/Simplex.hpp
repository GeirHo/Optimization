/*==============================================================================
SIMPLEX

The classical Nelder-Mead simplex algorithm [1] has been extended by Steven G.
Johnson to support bound constraints inspired by Box' approach [2]. The danger
of implementing bound constraints in this way is that you may collapse the
simplex into a lower-dimensional subspace. This may be avoided in the subspace
algorithm that uses Nelder-Mead on a sequence of subspaces [3], and also this
has been extended by Johnson to support bound constraints.

References:

[1] J. A. Nelder and R. Mead, "A simplex method for function minimization,"
    The Computer Journal Vol. 7, pp. 308-313, 1965
[2] M. J. Box, "A new method of constrained optimization and a comparison
    with other methods," The Computer Journal. Vol. 8, No. 1, pp. 42-52, 1965
[3] T. Rowan, "Functional Stability Analysis of Numerical Algorithms",
    Ph.D. thesis, Department of Computer Sciences, University of Texas at
    Austin, 1990.

Author and Copyright: Geir Horn, 2019
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_NON_LINEAR_SIMPLEX
#define OPTIMIZATION_NON_LINEAR_SIMPLEX

#include "../Variables.hpp"                   // Basic definitions

#include "NonLinear/Algorithms.hpp"           // Definition of the algorithms
#include "NonLinear/Objective.hpp"            // Objective function
#include "NonLinear/Optimizer.hpp"            // Optimizer interface
#include "NonLinear/Bounds.hpp"               // Variable domain bounds

namespace Optimization::NonLinear
{
/*==============================================================================

 Nelder-Mead simplex

==============================================================================*/

template<>
class Optimizer< Algorithm::Local::Simplex::NelderMead >
: virtual public NonLinear::Objective< Algorithm::Local::Simplex::NelderMead >,
  virtual public NonLinear::Bound,
	public NonLinear::OptimizerInterface
{
protected:

  // Define the algorithm used by the solver of this optimizer

  virtual Algorithm::ID GetAlgorithm( void ) override
	{ return Algorithm::Local::Simplex::NelderMead; }

  // The function to create the solver will also initialise the bounds
  // for the problem

	SolverPointer CreateSolver( Dimension NumberOfVariables,
														  Objective::Goal Direction) final
	{
		SolverPointer TheSolver =
									OptimizerInterface::CreateSolver( NumberOfVariables,
																										Direction);

	  Objective::SetObjective( TheSolver, Direction );
		SetBounds( TheSolver );

    return TheSolver;
	}

  // The constructor is protected to ensure that it only will be called by
	// derived classes defining the objective function and the bound function.
	// Its main purpose is to initialise the base classes. Note that the
	// virtual base classes must be initialised before the interface class.

	Optimizer( void )
	: Objective< Algorithm::Local::Simplex::NelderMead >(), Bound(),
	  OptimizerInterface()
	{}

public:

  virtual ~Optimizer( void )
  {}
};

/*==============================================================================

 Subspace simplex

==============================================================================*/
//
// Subspace simplex will initialise the solver exactly as the Nelder-Mead
// simplex, and thus there is no reason to duplicate the create solver
// function and it is readily inherited.

template<>
class Optimizer< Algorithm::Local::Simplex::Subspace >
: public Optimizer< Algorithm::Local::Simplex::NelderMead >
{
protected:

  // Define the algorithm used by the solver of this optimizer

  virtual Algorithm::ID GetAlgorithm( void ) override
	{ return Algorithm::Local::Simplex::Subspace; }

  // The constructor is protected to ensure that it only will be called by
	// derived classes defining the objective function and the bound function.
	// Its main purpose is to initialise the base classes. Note that the
	// virtual base classes must be initialised before the Nelder-Mead simplex.

	Optimizer( void )
	: Optimizer< Algorithm::Local::Simplex::NelderMead >()
	{}

public:

  virtual ~Optimizer( void )
  {}
};

}      // End name space Non Linear Optimization
#endif // OPTIMIZATION_NON_LINEAR_SIMPLEX
