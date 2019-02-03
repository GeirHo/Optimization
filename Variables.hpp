/*==============================================================================
Variables

A variable in an optimization problem is generally defined over a domain that 
can be numeric or non-numeric, and continuous or discrete. Although it is 
possible to define a variable class defined on a domain type covering all these
variations, the main intention of this optimization library is to provide a 
homogeneous wrapper for several optimization algorithms typically implemented 
in C for numerical functions taking real variables with double precision. 
Hence, for now, the variable type is defined to be a standard double. It is 
important to use the defined variable type instead of a standard double as 
its definition could change in the future.

As long as the variable type is a plain type and no class it is not possible 
to define safe conversions for it, and the built in conversions should be 
used (for now). 

Author and Copyright: Geir Horn, 2018
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_VARIABLES
#define OPTIMIZATION_VARIABLES

#include <vector>

namespace Optimization
{
using VariableType   = double;
using Variables      = std::vector< VariableType >;
using GradientVector = std::vector< VariableType >;
using Dimension      = typename Variables::size_type;

}      // End name space Optimization
#endif // OPTIMIZATION_VARIABLES
