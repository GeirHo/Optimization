/*==============================================================================
AGS

The AGS algorithm is derivative-free and employs the Hilbert curve to reduce
the source problem to the univariate one. Limitations of the machine arithmetic
do not allow to build a tight approximation for Hilbert when the space
dimension is greater than 5. This algorithm is no longer available in the
NLOpt package but could be provided as a separate library from the source
code available on GitHub [6].

The algorithm divides the univariate space into intervals, generating new points
by using posterior probabilities. On each trial AGS tries to evaluate the
constraints consequently one by one. If some constraint is violated at this
point, the next ones will not be evaluated. If all constraints are preserved,
i.e. the trial point is feasible, AGS will evaluate the objective.

There is a recent multi-objective revision available [4]. This method limits
the number of constraints to 5, and it is currently not available in NLOpt,
but it could be included as a special library here. The code is however
available on GitHub [5]

References:

[1] Yaroslav D. Sergeyev, Dmitri L. Markin: "An algorithm for solving global
    optimization problems with nonlinear constraints, Journal of Global
    Optimization, 7(4), pp 407–419, 1995
[2] Strongin R.G., Sergeyev Ya.D.: "Global optimization with non-convex
    constraints. Sequential and parallel algorithms". Kluwer Academic
    Publishers, Dordrecht, 2000
[3] Gergel V. and Lebedev I.: Heterogeneous Parallel Computations for Solving
    Global Optimization Problems. Proc. Comput. Science 66, pp. 53–62, 2015
[4] Sovrasov V.: Parallel Multi-Objective Optimization Method for Finding
    Complete Set of Weakly Efficient Solutions, Proceedings of the 3rd Ural
    Workshop on Parallel, Distributed, and Cloud Computing for Young Scientists,
    Yekaterinburg, Russia, October 19th, 2017
[5] https://github.com/sovrasov/multicriterial-go
[6] https://github.com/sovrasov/ags_nlp_solver

Author and Copyright: Geir Horn, 2019
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_NON_LINEAR_AGS
#define OPTIMIZATION_NON_LINEAR_AGS

#include <sstream>                            // For error reporting
#include <stdexcept>                          // For standard exceptions

#include "../Variables.hpp"                   // Basic definitions

#include "NonLinear/Algorithms.hpp"           // Definition of the algorithms
#include "NonLinear/Objective.hpp"            // Objective function
#include "NonLinear/Optimizer.hpp"            // Optimizer interface
#include "NonLinear/Bounds.hpp"               // Variable domain bounds
#include "NonLinear/Constraints.hpp"          // Constraint functions

namespace Optimization::NonLinear
{
template<>
class Optimizer< Algorithm::Global::AGS >
: virtual public NonLinear::Objective<Algorithm::Global::AGS>,
  virtual public NonLinear::Bound,
  virtual public NonLinear::InEqConstraints< Algorithm::Global::AGS >,
	public NonLinear::OptimizerInterface
{
private:

  // There is a tolerance parameter for the evaluation of the constraints

  double Tolerance;

protected:

  // The algorithm for this optimizer can be obtained in the standard way

  virtual Algorithm::ID GetAlgorithm( void ) override
	{ return Algorithm::Global::AGS; }

  // The solver can only be created if the number of variables is 5 or less
  // and an invalid argument exception is thrown if this is not the case.

  SolverPointer CreateSolver( Dimension NumberOfVariables,
														  Objective::Goal Direction) final
	{
    SolverPointer TheSolver;

    if ( NumberOfVariables < 6 )
      TheSolver = OptimizerInterface::CreateSolver( NumberOfVariables,
																										Direction);
    else
    {
      std::ostringstream ErrorMessage;

      ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
                   << "The AGS algorithm can only solve system with 5 or "
                   << "less variables, and it was attempted to be created "
                   << "for " << NumberOfVariables << " variables";

      throw std::invalid_argument( ErrorMessage.str() );
    }

    // The solver could be created so we may proceed to set the objective
    // function and the variable bounds.

    SetObjective( TheSolver, Direction );
		SetBounds( TheSolver );

    // Since the constraints are optional they may not be given and they are
    // set only if they have been provided.

    if ( NumberOfInEqConstraints() > 0 )
      SetInEqConstraints( TheSolver, Tolerance );

    return TheSolver;
	}

  // The constructor requires a value for the tolerance and it is protected
  // to ensure that it can only be called from derived classes. Since the
  // tolerance is required, the default constructor is deleted.

 	Optimizer( double ConstraintTolerance )
	: Objective(), Bound(), IndividualInEqConstraints(),
    OptimizerInterface(),
	  Tolerance( ConstraintTolerance )
	{}

  Optimizer( void ) = delete;

public:

  // The destructor is virtual to ensure that the sub-classes are properly
  // destroyed although it does nothing in particular for this class.

  virtual ~Optimizer( void )
  {}
};     // End class AGS specialisation
}      // End name space Non Linear Optimization
#endif // OPTIMIZATION_NON_LINEAR_AGS
