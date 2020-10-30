/*==============================================================================
Simple problem

This test problem creates a simple test problem showing how to use the
optimization interface. The sample problem used here is the same as the one
used in the NLOpt tutorial [1]. The problem is as follows:

minimize Sqrt( x[1] )

subject to

x[1] >= 0 (Bound constraint)
x[1] >= ( a[0] x[0] + b[0] )^3
x[1] >= ( a[1] x[0] + b[1] )^3

With given numerical values a[0] = 2, b[0] = 0, a[1] = 1 and b[1] = 1. Note
that the variable indices have been changed to i = 0... as opposed to i = 1..
used in standard mathematical notation.

References:
[1] https://nlopt.readthedocs.io/en/latest/NLopt_Tutorial/

Author and Copyright: Geir Horn, 2019
License: LGPL 3.0
==============================================================================*/

#include <cmath>             // Mathematical functions
#include <iostream>          // Status messages
#include <vector>            // Parameter vectors

#include "Objective.hpp"
#include "Constraints.hpp"
#include "NonLinear.hpp"

class SimpleProblem
: virtual public Optimization::Objective,
  virtual public Optimization::InEqConstraints
{
private:

  std::vector< Optimization::VariableType > a, b;

protected:

	Optimization::VariableType
	ObjectiveFunction( const Optimization::Variables & x ) override
	{ return std::sqrt( x[1] );	}

	// The constructor defines the constraints. Note that the bound constraint is
	// given as a normal constraint because it only bounds the variable in one
	// direction, and there are no bounds for the x[0] variable.
	//
	// The constraint expressions are defined as lambda expressions, which
	// requires a certain 'decoration' of the constraint expression. If many
	// constraints are to be defined this 'decoration' may of course be wrapped
	// up in a macro so that one could write only the expression and have it
	// expanded as the full version.
	//
	// This approach will always work, but if an algorithms supporting vector
	// constraints could be guaranteed, then one could define all constraints
	// in one vector function and avoid the evaluation of multiple constraint
	// functions.

public:

	SimpleProblem( void )
	: Objective(), InEqConstraints(),
	  a({2,1}), b({0,1})
	{
		InEqConstraints::Add(
			[]( const Optimization::Variables & x )->Optimization::VariableType{
				return -x[1];
			} );

		InEqConstraints::Add(
			[this]( const Optimization::Variables & x )->Optimization::VariableType{
				return std::pow( a[0]*x[0] + b[0], 3) - x[1];
			} );

		InEqConstraints::Add(
			[this]( const Optimization::Variables & x )->Optimization::VariableType{
				return std::pow( a[1]*x[0] + b[1], 3) - x[1];
			} );
	}

	// The destructor is virtual to ensure that all base classes are properly
	// destructed.

	virtual ~SimpleProblem( void )
	{}
};

