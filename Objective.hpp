/*==============================================================================
Objective function

The abstract objective class defines the real function to optimize taking a
vector of real values as argument values and returning a real number for the
objective function at that point.

The abstract gradient class defines a gradient function that is required by
some of the optimization algorithms. It takes a vector of real values and
returns a vector of real values of the same size as the argument vector
representing the gradient at the evaluation point.

Author and Copyright: Geir Horn, 2018
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMMIZATION_OBJECTIVE
#define OPTIMMIZATION_OBJECTIVE

#include <vector>
#include "Variables.hpp"

namespace Optimization
{

/*==============================================================================

 Objective function base class

==============================================================================*/

class Objective
{
public:

	// Even though the standard optimization is to minimize, both minimization
	// and maximization are possible, and the direction can in some cases be set.

	enum class Goal
	{
		Minimize,
		Maximize
	};

protected:

	virtual VariableType
	ObjectiveFunction( const Variables & VariableValues ) = 0;

	// The constructor is protected to prevent direct construction of this
	// class although it does not do anything.

	Objective( void )
	{}

public:

	// The virtual destructor is defined to allow derived classes to destruct
	// properly

	virtual ~Objective( void )
	{}
};

/*==============================================================================

 Gradient function base class

==============================================================================*/

class ObjectiveGradient : virtual public Objective
{
protected:

	virtual GradientVector
	GradientFunction( const Variables & VariableValues ) = 0;

	ObjectiveGradient( void )
	: Objective()
	{}

public:

	// Again there is a virtual destructor to allow derived classes to be
	// correctly destructed

	virtual ~ObjectiveGradient( void )
	{ }
};

}      // end name space Optimization
#endif // OPTIMMIZATION_OBJECTIVE
