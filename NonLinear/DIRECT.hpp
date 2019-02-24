/*==============================================================================
DIRECT: DIviding RECTangles

The DIviding RECTangles algorithm for global optimization [1] is based on
systematic division of the search domain into smaller and smaller
hyperrectangles. The "locally biased" variant [2] makes the algorithm "more
biased towards local search" so that it is more efficient for functions without
too many local minima. The NLopt library implements several variants of these
algorithms, and each of them is given as a specialised class here.

References:

[1] D. R. Jones, C. D. Perttunen, and B. E. Stuckmann: "Lipschitzian
    optimization without the lipschitz constant," J. Optimization Theory and
    Applications, vol. 79, p. 157 (1993).
[2] J. M. Gablonsky and C. T. Kelley, "A locally-biased form of the DIRECT
    algorithm," J. Global Optimization, vol. 21 (1), p. 27-37 (2001).

Author and Copyright: Geir Horn, 2018
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_NON_LINEAR_DIRECT
#define OPTIMIZATION_NON_LINEAR_DIRECT

#include "../Variables.hpp"                   // Basic definitions

#include "NonLinear/Algorithms.hpp"           // Definition of the algorithms
#include "NonLinear/Objective.hpp"            // Objective function
#include "NonLinear/Optimizer.hpp"            // Optimizer interface
#include "NonLinear/Bounds.hpp"               // Variable domain bounds
#include "NonLinear/Constraints.hpp"          // Constraint functions


namespace Optimization::NonLinear
{
/*==============================================================================

 DIRECT Standard version

==============================================================================*/
//
// This is the standard version re-implemented for the NLopt library.

template<>
class Optimizer< Algorithm::Global::DIRECT::Standard >
: virtual public NonLinear::Objective< Algorithm::Global::DIRECT::Standard >,
  virtual public NonLinear::Bound,
	public NonLinear::OptimizerInterface
{
protected:

	// All the variants will fundamentally initialize the solver in the same way
	// at the exception of the algorithm passed, hence there is a simple
	// interface function to provide the correct version of the algorithm.

	virtual Algorithm::ID GetAlgorithm( void ) override
	{ return Algorithm::Global::DIRECT::Standard;	}

	// Creating the solver is in this case just allocating the solver and
	// setting the variable bounds since the algorithm does not support any other
	// constraints.

	SolverPointer
	CreateSolver( Dimension NumberOfVariables, Objective::Goal Direction) final
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

	// The destructor is public and virtual to ensure that all the polymorphic
	// classes are correctly destroyed.

public:

	virtual ~Optimizer( void )
	{}
};

/*==============================================================================

 DIRECT Unscaled

==============================================================================*/
//
// The standard algorithm gives all dimensions equal weight in the search
// procedure, but if the dimensions does not have equal weights, the unscaled
// version may work better. It is identical in implementation to the DIRECT
// standard algorithm and it is therefore based on the standard version

template<>
class Optimizer< Algorithm::Global::DIRECT::Unscaled >
: public Optimizer< Algorithm::Global::DIRECT::Standard >
{
protected:

	// Define the algorithm used by the solver of this optimizer

  virtual Algorithm::ID GetAlgorithm( void ) final
	{ return Algorithm::Global::DIRECT::Unscaled;	}

	// The constructor is protected to ensure that it only will be called by
	// derived classes defining the objective function and the bound function.
	// Its main purpose is to initialise the base classes. Note that the
	// virtual base classes must be initialised before the interface class.

	Optimizer( void )
	: Optimizer< Algorithm::Global::DIRECT::Standard >()
	{}

	// The destructor is public and virtual to ensure that all the polymorphic
	// classes are correctly destroyed.

public:

	virtual ~Optimizer( void )
	{}
};

/*==============================================================================

 DIRECT Original

==============================================================================*/
//
// The original FORTRAN implementation of the algorithm was provided by the
// authors of the local search version of DIRECT (Gablonsky and Kelly),
// and this is also available in NLopt, and the interface is identical to
// the ones for the DIRECT. According to the description of the algorithm
// in NLopt this variant "includes some support for arbitrary non-linear
// inequality constraints."

template<>
class Optimizer< Algorithm::Global::DIRECT::Original >
: virtual public NonLinear::Objective< Algorithm::Global::DIRECT::Original >,
  virtual public NonLinear::Bound,
	virtual public NonLinear::InEqConstraints< Algorithm::Global::DIRECT::Original >,
	public NonLinear::OptimizerInterface
{
private:

	// The inequality constraints require a tolerance to be set, and this is
	// required to be given to the constructor.

	double Tolerance;

protected:

	// Define the algorithm used by the solver of this optimizer

  virtual Algorithm::ID GetAlgorithm( void ) override
	{ return Algorithm::Global::DIRECT::Original;	}

	// In this case the function to create the solver must be overloaded to
	// initialize also the constraints.

	SolverPointer CreateSolver( Dimension NumberOfVariables,
														  Objective::Goal Direction) final
	{
		SolverPointer TheSolver =
									OptimizerInterface::CreateSolver( NumberOfVariables,
																										Direction);

	  Objective::SetObjective( TheSolver, Direction );
		SetBounds( TheSolver );
    if ( NumberOfInEqConstraints() > 0 )
		   SetInEqConstraints( TheSolver, Tolerance );

    return TheSolver;
	}

	// The constructor is protected to ensure that it only will be called by
	// derived classes defining the objective function and the bound function.
	// Its main purpose is to initialise the base classes. Note that the
	// virtual base classes must be initialised before the interface class.

	Optimizer( double ConstraintTolerance )
	: Objective(), Bound(), InEqConstraints(), OptimizerInterface(),
	  Tolerance( ConstraintTolerance )
	{}

  // The default constructor is not to be used

  Optimizer( void )	= delete;

	// The destructor is public and virtual to ensure that all the polymorphic
	// classes are correctly destroyed.

public:

	virtual ~Optimizer( void )
	{}
};

/*==============================================================================

 DIRECT Local

==============================================================================*/
//
// DIviding RECTangles with local search is similar to the DIRECT class
// except for the algorithm is biased towards local search and therefore
// faster for function not having too many local optima. The implementation is
// based on the standard version of DIRECT

template<>
class Optimizer< Algorithm::Global::DIRECT::Local::Standard >
: public Optimizer< Algorithm::Global::DIRECT::Standard >
{
protected:

	// Define the algorithm used by the solver of this optimizer

  virtual Algorithm::ID GetAlgorithm( void ) override
	{ return Algorithm::Global::DIRECT::Local::Standard;	}

	// The constructor is protected to ensure that it only will be called by
	// derived classes defining the objective function and the bound function.
	// Its main purpose is to initialise the base classes. Note that the
	// virtual base classes must be initialised before the interface class.

	Optimizer( void )
	: Optimizer< Algorithm::Global::DIRECT::Standard >()
	{}

	// The destructor is public and virtual to ensure that all the polymorphic
	// classes are correctly destroyed.

public:

	virtual ~Optimizer( void )
	{}
};

/*==============================================================================

 DIRECT Local Original

==============================================================================*/
//
// It has also the original implementation of the algorithm, and this supports
// the inequality constraints like the original DIRECT algorithm. It is
// therefore based on the original implementation of the DIRECT algorithm.

template<>
class Optimizer< Algorithm::Global::DIRECT::Local::Original >
: public Optimizer< Algorithm::Global::DIRECT::Original >
{
protected:

	// Define the algorithm used by the solver of this optimizer

  virtual Algorithm::ID GetAlgorithm( void ) final
	{ return Algorithm::Global::DIRECT::Local::Original; }

	// The constructor requires the tolerance to use for the inequality
	// constraints and passes this to the constructor of the original
	// implementation of the DIRECT class

	Optimizer( double ConstraintTolerance  )
	: Optimizer< Algorithm::Global::DIRECT::Original >( ConstraintTolerance )
	{}

	// The destructor is public and virtual to ensure that all the polymorphic
	// classes are correctly destroyed.

public:

	virtual ~Optimizer( void )
	{}
};

/*==============================================================================

 DIRECT Local Randomized

==============================================================================*/
//
// The randomized version uses randomization to help decide which dimension
// to halve next in the case of near-ties. The implementation is again based
// on the interface for the standard DIRECT algorithm.

template<>
class Optimizer< Algorithm::Global::DIRECT::Local::Randomized >
: public Optimizer< Algorithm::Global::DIRECT::Standard >
{
protected:

	// Define the algorithm used by the solver of this optimizer

  virtual Algorithm::ID GetAlgorithm( void ) final
	{ return Algorithm::Global::DIRECT::Local::Randomized; }

	// The constructor is protected to ensure that it only will be called by
	// derived classes defining the objective function and the bound function.
	// Its main purpose is to initialise the base classes. Note that the
	// virtual base classes must be initialised before the interface class.

	Optimizer( void )
	: Optimizer< Algorithm::Global::DIRECT::Standard >()
	{}

	// The destructor is public and virtual to ensure that all the polymorphic
	// classes are correctly destroyed.

public:

	virtual ~Optimizer( void )
	{}
};

/*==============================================================================

 DIRECT Local Standard Unscaled

==============================================================================*/
//
// Similar to the DIRECT algorithm it has also an unscaled version for the
// case where there are huge differences in the scale of the optimization
// dimensions.

template<>
class Optimizer< Algorithm::Global::DIRECT::Local::Unscaled::Standard >
: public Optimizer< Algorithm::Global::DIRECT::Standard >
{
protected:

	// Define the algorithm used by the solver of this optimizer

  virtual Algorithm::ID GetAlgorithm( void ) final
	{ return Algorithm::Global::DIRECT::Local::Unscaled::Standard; }

	// The constructor is protected to ensure that it only will be called by
	// derived classes defining the objective function and the bound function.
	// Its main purpose is to initialise the base classes. Note that the
	// virtual base classes must be initialised before the interface class.

	Optimizer( void )
	: Optimizer< Algorithm::Global::DIRECT::Standard >()
	{}

	// The destructor is public and virtual to ensure that all the polymorphic
	// classes are correctly destroyed.

public:

	virtual ~Optimizer( void )
	{}
};

/*==============================================================================

 DIRECT Local Randomized Unscaled

==============================================================================*/
//
// There is also a randomized version of the unscaled variant reusing the
// interface for the standard DIRECT algorithm.

template<>
class Optimizer< Algorithm::Global::DIRECT::Local::Unscaled::Randomized >
: public Optimizer< Algorithm::Global::DIRECT::Standard >
{
protected:

	// Define the algorithm used by the solver of this optimizer

  virtual Algorithm::ID GetAlgorithm( void ) final
	{ return Algorithm::Global::DIRECT::Local::Unscaled::Randomized; }

	// The constructor is protected to ensure that it only will be called by
	// derived classes defining the objective function and the bound function.
	// Its main purpose is to initialise the base classes. Note that the
	// virtual base classes must be initialised before the interface class.

	Optimizer( void )
	: Optimizer< Algorithm::Global::DIRECT::Standard >()
	{}

	// The destructor is public and virtual to ensure that all the polymorphic
	// classes are correctly destroyed.

public:

	virtual ~Optimizer( void )
	{}
};


}      // Name space Optimization non-linear
#endif // OPTIMIZATION_NON_LINEAR_DIRECT

