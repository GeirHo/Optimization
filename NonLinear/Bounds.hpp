/*==============================================================================
Bounds

Most of the algorithms support bounds for the variable domains. The bounds are 
basically intervals, and they are therefore defined in terms of the boost 
intervals, and the bounds constraint function returns a vector of intervals, 
one for each variable of the problem.

Author and Copyright: Geir Horn, 2018
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_NON_LINEAR_BOUNDS
#define OPTIMIZATION_NON_LINEAR_BOUNDS

#include <vector>											        // For variables and values
#include <boost/numeric/interval.hpp>         // For variable domains (ranges)
#include <nlopt.h>                            // The C-style interface

#include "Variables.hpp"                      // Variable definitions
#include "Algorithms.hpp"                     // Algorithm definitions

namespace Optimization::NonLinear
{

class Bound
{
protected:
	
	using Interval = boost::numeric::interval< VariableType >;
	
	virtual std::vector< Interval > BoundConstraints( void ) = 0;
	
	// Normally the above function is only called indirectly via the function to 
	// set the bounds for a given solver. The NLopt interface for the bounds is 
	// to set them separately, and hence they must be taken from the intervals 
	// returned by the constraint function.
	
	inline void SetBounds( SolverPointer Solver )
	{
		std::vector< VariableType > Lower, Upper;
		
		for ( const Interval & VariableRange : BoundConstraints() )
		{
			Lower.push_back( VariableRange.lower() );
			Upper.push_back( VariableRange.upper() );
		}
		
		nlopt_set_lower_bounds( Solver, Lower.data() );
		nlopt_set_upper_bounds( Solver, Upper.data() );
	}

public:
	
	virtual ~Bound( void )
	{ }
};

}      // Name space Optimization non-linear
#endif // OPTIMIZATION_NON_LINEAR_BOUNDS
