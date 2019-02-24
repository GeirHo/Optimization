/*==============================================================================
Evolutionary

This file implements the specialisations for the algorithms that uses some
kind of evolutionary strategy in the serach for a solution. The improved
stochastic ranking evolution strategy [1] is based on a combination of a
mutation rule with a log-normal step-size update and exponential smoothing, and
differential variation like a Nelder–Mead update rule. The fitness ranking
is simply via the objective function for problems without nonlinear
constraints, but when nonlinear constraints are included a stochastic ranking
is employed. Even though the algorithm is claimed to be global, there is no
proof of global convergence and hence it has been classified as a local
algorithm in this interface.

The second algorithm implemented is a modified evolutionary algorithm [2,3],
only supporting bound constraints.

References:

[1] Thomas Philip Runarsson and Xin Yao, "Search biases in constrained
    evolutionary optimization," IEEE Trans. on Systems, Man, and Cybernetics
    Part C: Applications and Reviews, vol. 35 (no. 2), pp. 233-243 (2005)
[2] C. H. da Silva Santos, M. S. Gonçalves, and H. E. Hernandez-Figueroa,
    "Designing Novel Photonic Devices by Bio-Inspired Computing,"
    IEEE Photonics Technology Letters 22 (15), pp. 1177–1179, 2010
[3] C. H. da Silva Santos, "Parallel and Bio-Inspired Computing Applied to
    Analyze Microwave and Photonic Metamaterial Strucutures," Ph.D. thesis,
    University of Campinas, 2010

Author and Copyright: Geir Horn, 2019
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_NON_LINEAR_EVOLUTIONARY
#define OPTIMIZATION_NON_LINEAR_EVOLUTIONARY

#include "../Variables.hpp"                   // Basic definitions

#include "NonLinear/Algorithms.hpp"           // Definition of the algorithms
#include "NonLinear/Objective.hpp"            // Objective function
#include "NonLinear/Optimizer.hpp"            // Optimizer interface
#include "NonLinear/Bounds.hpp"               // Variable domain bounds
#include "NonLinear/Constraints.hpp"          // Constraint functions

namespace Optimization::NonLinear
{
/*==============================================================================

 ISRES: Improved Stochastic Ranking Evolution Strategy

==============================================================================*/

template<>
class Optimizer< Algorithm::Local::Evolutionary >
: virtual public NonLinear::Objective< Algorithm::Local::Evolutionary >,
  virtual public NonLinear::Bound,
  virtual public NonLinear::InEqConstraints< Algorithm::Local::Evolutionary >,
  virtual public NonLinear::EqConstraints< Algorithm::Local::Evolutionary >,
	public NonLinear::OptimizerInterface
{
private:

  // The method has two parameters: The tolerance used to evaluate the
  // constraints and the population size used in the evolution.

  double       Tolerance;
  unsigned int PopulationSize;

protected:

  // The algorithm used by the solver

  virtual Algorithm::ID GetAlgorithm( void ) override
	{ return Algorithm::Local::Evolutionary; }

  // The function to create the solver also sets the constraints that must
  // have been defined prior to creating the solver.

  SolverPointer CreateSolver( Dimension NumberOfVariables,
													    Objective::Goal Direction) final
	{
		SolverPointer TheSolver =
									OptimizerInterface::CreateSolver( NumberOfVariables,
																										Direction);

    Objective::SetObjective( TheSolver, Direction );
		SetBounds( TheSolver );

    // Since the constraints are optional they may not be given and they are
    // set only if they have been provided.

    if ( NumberOfInEqConstraints() > 0 )
      InEqConstraints::SetInEqConstraints( TheSolver, Tolerance );

    if ( NumberOfEqConstraints() > 0 )
      EqConstraints::SetEqConstraints( TheSolver, Tolerance );

    // Finally, if the population size has been set, then it will be
    // given to the solver.

    if ( PopulationSize > 0 )
      nlopt_set_population( TheSolver, PopulationSize );

    return TheSolver;
	}

  // The constructor takes the tolerance value and optionally the population
  // size. Since the tolerance must be given, the default constructor is
  // deleted.

  Optimizer( double ConstraintTolerance, unsigned int SizeOfPopulation = 0 )
	: Objective(), Bound(), InEqConstraints(), EqConstraints(),
    OptimizerInterface(),
	  Tolerance( ConstraintTolerance ), PopulationSize( SizeOfPopulation )
	{}

  Optimizer( void ) = delete;

public:

  // The destructor is virtual to ensure that the sub-classes are properly
  // destroyed although it does nothing in particular for this class.

  virtual ~Optimizer( void )
  {}
};

/*==============================================================================

 ESCH: Evolutionary algorithm

==============================================================================*/

template<>
class Optimizer< Algorithm::Global::Evolutionary >
: virtual public NonLinear::Objective< Algorithm::Global::Evolutionary >,
  virtual public NonLinear::Bound,
	public NonLinear::OptimizerInterface
{
protected:

  // Define the algorithm used by the solver of this optimizer

  virtual Algorithm::ID GetAlgorithm( void ) override
	{ return Algorithm::Global::Evolutionary; }

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
	: Objective(), Bound(), OptimizerInterface()
	{}

public:

  virtual ~Optimizer( void )
  {}
};

}      // End name space Non Linear Optimization
#endif // OPTIMIZATION_NON_LINEAR_EVOLUTIONARY
