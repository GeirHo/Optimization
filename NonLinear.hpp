/*==============================================================================
Non-Linear optimization

This class uses the NLOpt library [1] of algorithms to do the required
optimisation by wrapping the necessary interface classes since the NLopt
library is fundamentally C-style and the C++ interface to NLopt is unfortunately
not complete and this interface replaces the automatically generated NLopt
C++ interface directly using the C-style interface of NLopt. It should be noted
that if new algorithms are added to NLopt, they must be manually implemented
here.

The purpose of this redesigned C++ interface is to prevent the user from setting
up optimisation problems that will fail on execution. For instance, with the
automatically generated interface it is perfectly possible not to provide bounds
on the search domain even if an algorithm requires that; or one may fail to
provide the gradient functions for algorithms requiring them. Such code will
simply not compile with this interface.

The interface is also striving to improve the readability of the code, albeit
at the expense of some more typing since simplicity and readability is, in
general, two indications of better and more maintainable code.

Finally, the interface is designed to be mix and match so that the amount of
new code that needs to be added for defining a new algorithm should be minimal,
and there should virtually be no code duplication.

The original intention was to provide the interface as a single file, but
when the file length passed 2000 lines of code, it did no longer help the
understanding to keep it as one file and it was split into files for the
individual sub-classes or main algorithms of which all variants are kept in
the same header. The headers are included here, so one may still include only
this file and get all algorithm classes.

References:

[1] Steven G. Johnson: The NLopt nonlinear-optimization package,
    http://ab-initio.mit.edu/nlopt

Author and Copyright: Geir Horn, 2018
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_NON_LINEAR
#define OPTIMIZATION_NON_LINEAR

// -----------------------------------------------------------------------------
// Core interface
// -----------------------------------------------------------------------------

#include "NonLinear/Algorithms.hpp"           // Definition of the algorithms
#include "NonLinear/Objective.hpp"            // Objective function
#include "NonLinear/Optimizer.hpp"            // Optimizer interface
#include "NonLinear/Bounds.hpp"               // Variable domain bounds
#include "NonLinear/Constraints.hpp"          // Constraint functions

// -----------------------------------------------------------------------------
// Optimisers for various algorithms
// -----------------------------------------------------------------------------

#include "NonLinear/DIRECT.hpp"                  // DIviding RECTangles
#include "NonLinear/ControlledRandomSearch.hpp"  // CRS algorithm
#include "NonLinear/MultiLevelSingleLinkage.hpp" // MLSL algorithm
#include "NonLinear/StoGo.hpp"                   // Stochastic Global opt.

#endif // OPTIMIZATION_NON_LINEAR
