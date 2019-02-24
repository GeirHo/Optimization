/*==============================================================================
Principal axis

The principal axis method is fundamentally for unconstrained problems.
Steven G. Johnson implements support for bound constraints by returning
infinity for the object function if the argument is outside of the bounded
region. Naturally, this leads to slow convergence, and bound constraints are
therefore not supported by the implemented specialisation for this algorithm.
Johnson recommends one of the approximation algorithms for bounded problems.

References:

[1] Richard Brent, Algorithms for Minimization without Derivatives,
    Prentice-Hall, 1972

Author and Copyright: Geir Horn, 2019
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_NON_LINEAR_PRINCIPAL_AXIS
#define OPTIMIZATION_NON_LINEAR_PRINCIPAL_AXIS

#include "../Variables.hpp"                   // Basic definitions

#include "NonLinear/Algorithms.hpp"           // Definition of the algorithms
#include "NonLinear/Objective.hpp"            // Objective function
#include "NonLinear/Optimizer.hpp"

namespace Optimization::NonLinear
{
template<>
class Optimizer< Algorithm::Local::PrincipalAxis >
: virtual public NonLinear::Objective< Algorithm::Local::PrincipalAxis >,
	public NonLinear::OptimizerInterface
{
protected:

  // Define the algorithm used by the solver of this optimizer

  virtual Algorithm::ID GetAlgorithm( void ) override
	{ return Algorithm::Local::PrincipalAxis; }

  // The function to create the solver is trivial in this case as only
  // the objective function can be set.

	SolverPointer CreateSolver( Dimension NumberOfVariables,
														  Objective::Goal Direction) final
	{
		SolverPointer TheSolver =
									OptimizerInterface::CreateSolver( NumberOfVariables,
																										Direction);

	  Objective::SetObjective( TheSolver, Direction );

    return TheSolver;
	}

  // The constructor is protected to ensure that it only will be called by
	// derived classes defining the objective function. Note that the
	// virtual base classes must be initialised before the interface class.

	Optimizer( void )
	: Objective< Algorithm::Local::PrincipalAxis >(), OptimizerInterface()
	{}

public:

  virtual ~Optimizer( void )
  {}
};


}      // End name space Non Linear Optimization
#endif // OPTIMIZATION_NON_LINEAR_PRINCIPAL_AXIS
