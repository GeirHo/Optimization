/*==============================================================================
Stochastic Global optimisation (StoGo)

There is no published paper for this method, but there are some references 
cached as part of the NLOpt documentation. The algorithm comes in two versions:
The standard algorithm and one randomized.

Author and Copyright: Geir Horn, 2018
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_NON_LINEAR_STOGO
#define OPTIMIZATION_NON_LINEAR_STOGO

#include "../Variables.hpp"                   // Basic definitions

#include "NonLinear/Algorithms.hpp"           // Definition of the algorithms
#include "NonLinear/Objective.hpp"            // Objective function
#include "NonLinear/Optimizer.hpp"            // Optimizer interface
#include "NonLinear/Bounds.hpp"               // Variable domain bounds


namespace Optimization::NonLinear
{
/*==============================================================================

 Standard Stochastic Global optimisation

==============================================================================*/

template<>
class Optimizer< Algorithm::Global::StoGo::Standard >
: virtual public NonLinear::ObjectiveGradient,
  virtual public NonLinear::Bound,
	public OptimizerInterface
{	
protected:
	
	// The algorithm is readily defined.
	
	virtual Algorithm::ID GetAlgorithm( void ) override
	{ 
		return Algorithm::Global::StoGo::Standard;	
	}
	
	// Creating the solver is in this case just allocating the solver and 
	// setting the variable bounds since the algorithm does not support any other
	// constraints.
	
	SolverPointer 
	CreateSolver( Dimension NumberOfVariables, Objective::Goal Direction) final
	{
		SolverPointer TheSolver = 
									OptimizerInterface::CreateSolver( NumberOfVariables, 
																										Direction);
		
	  SetObjective( TheSolver, Direction );
		SetBounds( TheSolver );
	
		return TheSolver;
	}
	
	// The constructor only initialises the base classes
	
	Optimizer( void )
	: Objective(), Bound(), OptimizerInterface()
	{}
	
	// The destructor is public and virtual to ensure that all the polymorphic
	// classes are correctly destroyed.
	
public:
	
	virtual ~Optimizer( void )
	{}
};

/*==============================================================================

 Randomized Stochastic Global optimisation

==============================================================================*/
//
// The randomized variant simply changes the algorithm function as the other 
// functionality remains the same as for the standard version, and it is 
// therefore based on the standard version.

template<>
class Optimizer< Algorithm::Global::StoGo::Randomized >
: virtual public NonLinear::ObjectiveGradient,
  virtual public NonLinear::Bound,
  public Optimizer< Algorithm::Global::StoGo::Standard >
{
protected:
	
	virtual Algorithm::ID GetAlgorithm( void ) override
	{ 
		return Algorithm::Global::StoGo::Randomized;	
	}
	
	Optimizer( void )
	: ObjectiveGradient(), Bound(), 
	  Optimizer< Algorithm::Global::StoGo::Standard >()
	{}
	
public:
	
	virtual ~Optimizer( void )
	{}
};


}
#endif // OPTIMIZATION_NON_LINEAR_STOGO

