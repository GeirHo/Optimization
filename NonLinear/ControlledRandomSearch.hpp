/*==============================================================================
Controlled Random Search

The CRS algorithms are sometimes compared to genetic algorithms, in that
they start with a random "population" of points, and randomly "evolve"
these points by heuristic rules. In this case, the "evolution" somewhat
resembles a randomized Nelder-Mead algorithm. The NLotp implemented version is
based on [1], but the original algorithm was described in [2,3].

References:

[1] P. Kaelo and M. M. Ali, "Some variants of the controlled random search
    algorithm for global optimization," J. Optim. Theory Appl. 130 (2),
    pp. 253-264 (2006)
[2] W. L. Price, "A controlled random search procedure for global optimization"
    in Towards Global Optimization 2, p. 71-84 edited by L. C. W. Dixon and
    G. P. Szego (North-Holland Press, Amsterdam, 1978).
[3] W. L. Price, "Global optimization by controlled random search," J. Optim.
    Theory Appl. 40 (3), p. 333-348 (1983).

Author and Copyright: Geir Horn, 2018
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_NON_LINEAR_CRS
#define OPTIMIZATION_NON_LINEAR_CRS

#include "../Variables.hpp"
#include "NonLinear/Algorithms.hpp"           // Definition of the algorithms
#include "NonLinear/Objective.hpp"            // Objective function
#include "NonLinear/Optimizer.hpp"            // Optimizer interface
#include "NonLinear/Bounds.hpp"               // Variable domain bounds

namespace Optimization::NonLinear
{

template<>
class Optimizer< Algorithm::Global::ControlledRandomSearch >
: virtual public NonLinear::Objective< Algorithm::Global::ControlledRandomSearch >,
  virtual public NonLinear::Bound,
	public NonLinear::OptimizerInterface
{
private:

	// The initial population size for the search algorithm is by default set to
	// 10 * (n+1) where n is the number of variables in the problem. It can be
	// set to any value and it is therefore a variable to remember the choice.

	unsigned int InitialPopulationSize;

	// Standard short hand for the objective base class

	using Objective =
	NonLinear::Objective< Algorithm::Global::ControlledRandomSearch >;

protected:

	// There is a utility function to set this size. It must be at least n+1
	// where n is the population size, and the validity of the set population
	// size will be tested when the solver is created.

	inline void SetPopulationSize( unsigned int Value )
	{
		InitialPopulationSize = Value;
	}

	// All the variants will fundamentally initialize the solver in the same way
	// at the exception of the algorithm passed, hence there is a simple
	// interface function to provide the correct version of the algorithm.

	virtual Algorithm::ID GetAlgorithm( void ) final
	{
		return Algorithm::Global::ControlledRandomSearch;
	}

	// Creating the solver is in this case just allocating the solver and
	// setting the variable bounds since the algorithm does not support any other
	// constraints.

	SolverPointer
	CreateSolver( Dimension NumberOfVariables, Objective::Goal Direction) final
	{
		SolverPointer TheSolver =
									OptimizerInterface::CreateSolver( NumberOfVariables,
																										Direction );

	  Objective::SetObjective( TheSolver, Direction );
		SetBounds( TheSolver );

		// Testing and setting the initial population size

		if ( InitialPopulationSize > NumberOfVariables + 1 )
			nlopt_set_population( TheSolver, InitialPopulationSize );

		return TheSolver;
	}

	// The constructor is protected to ensure that it only will be called by
	// derived classes defining the objective function and the bound function.
	// Its main purpose is to initialise the base classes. Note that the
	// virtual base classes must be initialised before the interface class.
	//
	// The initial population size is by default set to zero.

	Optimizer( void )
	: Objective(), Bound(), OptimizerInterface(),
	  InitialPopulationSize( 0 )
	{}

	// The destructor is public and virtual to ensure that all the polymorphic
	// classes are correctly destroyed.

public:

	virtual ~Optimizer( void )
	{}
};

}      // Name space Optimization non-linear
#endif // OPTIMIZATION_NON_LINEAR_CRS

