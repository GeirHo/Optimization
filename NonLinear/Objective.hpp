/*==============================================================================
Objective

This header provides a C-style wrapper for the indirection function for the 
objective function. It is based on the generic objective function, and provides 
the objective function and the definition for the objective gradient function.

The standard objective function can be used, but it is extended to set the 
objective function for an optimiser object and to provide a indirection 
mapper confirming to the format expected by the C-style objective function 
used by NLopt. It inherits the objective function as a virtual base class 
to ensure that there is only one objective.

The gradient is a tricky point as it is expected by default to the objective 
function for all algorithms, even if the algorithm does not use the
gradient values. Thus if only the plain objective function is used the
gradient value vector should be an empty vector. To force that the gradient
objective class is abstract if the gradient function is not implemented, 
a secondary computation function is defined.

Author and Copyright: Geir Horn, 2018
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_NON_LINEAR_OBJECTIVE
#define OPTIMIZATION_NON_LINEAR_OBJECTIVE

#include <vector>											        // For variables and values
#include <sstream>                            // For error reporting
#include <stdexcept>                          // For standard exceptions

#include "../Variables.hpp"                   // Basic definitions
#include "../Objective.hpp"                   // Objective function
#include "NonLinear/Algorithms.hpp"           // Non-linear definitions

#include <nlopt.h>                            // The C-style interface

namespace Optimization::NonLinear
{

class Objective : virtual public Optimization::Objective
{
public:
	
	// Even though the standard optimization is to minimize, both minimization 
	// and maximization are possible, and the direction can be passed to the 
	// function creating the solver.
	
	enum class Goal
	{
		Minimize,
		Maximize
	};

protected:
	
	// The gradient computation function must be overloaded by algorithms 
	// requiring gradient values to be computed. If the default version is 
	// called, it is an indication that the derived problem definition has 
	// not been correctly done.
	
	virtual GradientVector 
	ComputeGradient( const Variables & VariableValues )
	{
		std::ostringstream ErrorMessage;
		
		ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
		             << "The algorithm used requires the gradient function "
								 << " to be defined. However, the empty gradient function"
								 << " was called and it should have been overloaded";
								 
	  throw std::logic_error( ErrorMessage.str() );
	}
	
	// The mapper function takes a vector of variable values and evaluates the 
	// objective function for these. It is assumed that the Parameters is only 
	// a pointer to this objective class, and a static cast to this has to be 
	// used since only static casts are supported from void pointers.
	
	static double IndirectionMapper( 
		unsigned int Size,       const double * ArgumentValues, 
		double * GradientValues, void * Parameters)
	{
		// The parameters should be a pointer to this objective class.
		
		Objective * This = reinterpret_cast< Objective * >( Parameters );
		
		// Setting the variable values based on the given arguments.
		
		Variables VariableValues( Size );
		
		for ( Dimension i = 0; i < Size; i++ )
			VariableValues[i] = ArgumentValues[i];
		
		// Compute the gradient values if the gradient pointer is not null
		
		if ( GradientValues != nullptr )
		{
			GradientVector
			Gradient( This->ComputeGradient( VariableValues ) );
			
			for ( Dimension i = 0; i < Size; i++ )
				GradientValues[i] = Gradient[i];
		}
		
		// Compute and return the objective function value 
		
		return This->ObjectiveFunction( VariableValues );
	}
	
	// There are functions to set the objective function for a solver, both 
	// as a standard minimization objective or as a maximization objective.
	
	inline void SetObjective( SolverPointer Solver, 
														Goal Direction = Goal::Minimize )
	{
		switch( Direction )
		{
			case Goal::Minimize:
				nlopt_set_min_objective( Solver, &IndirectionMapper, this );
				break;
			case Goal::Maximize:
				nlopt_set_max_objective( Solver, &IndirectionMapper, this );
				break;
		}
	}
	
public:
	
	// A virtual destructor is needed to ensure that derived objects are 
	// properly closed.
	
	virtual ~Objective( void )
	{}
};

// -----------------------------------------------------------------------------
// Object Gradient 
// -----------------------------------------------------------------------------
//
// The objective gradient extends the above objective function class with a 
// real method for computing the objective function gradient, and uses this 
// to return values when the indirection mapper calls the compute gradient 
// method.

class ObjectiveGradient
: virtual public Objective,
  virtual public Optimization::ObjectiveGradient
{
protected:
	
	// The compute gradient function should not be overloaded by any other class
	// and it is therefore declared final.
	
	virtual GradientVector 
	ComputeGradient( const Variables & VariableValues ) final
	{
		return GradientFunction( VariableValues );
	}
	
public:
	
	virtual ~ObjectiveGradient( void )
	{}
};

}      // Name space Optimization non-linear
#endif // OPTIMIZATION_NON_LINEAR_OBJECTIVE
