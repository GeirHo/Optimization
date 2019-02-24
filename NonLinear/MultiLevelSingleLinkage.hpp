/*==============================================================================
Multi-Level Single-Linkage

The Multi-Level Single-Linkage algorithm [1] selects a set of random staring
points and then uses another algorithm to search for a local optimum around each
starting point. Both the number of initial points and the local algorithm must
be given as parameters to the constructor, and both may be changed on subsequent
invocations of the algorithm.

The NLOpt documentation makes a point in suggesting that the stopping tolerances
for the objective function and the relative variable value change could (and
should) be set relatively large in the beginning and then once a solution has
been identified, one may run a second search starting from the approximative
optimum identified, but with much smaller tolerances. It is therefore
possible to set the tolerances for the local search.

The sub-algorithm must use the same objective function as the top level
algorithm. In other words, if the objective function does not provide a
gradient, the sub-algorithm cannot require a gradient based algorithm.
The optimizer class will throw an error if this situation happens.

There are two main variants of this algorithm. One that uses randomized
starting points, and one that uses a Sobol sequence [2] as a low-discrepancy
sequence [3], which arguably improves the conversion rate [4].
The low-discrepancy variants are identical with respect to implementation and
management of the sub-algorithms, and differ from the multi-level single linkage
variants only in the algorithm used at the top level.

References:

[1] A. H. G. Rinnooy Kan and G. T. Timmer, "Stochastic global optimization
    methods," Mathematical Programming, vol. 39, pp. 27-78, 1987
[2] https://en.wikipedia.org/wiki/Sobol_sequence
[3] https://en.wikipedia.org/wiki/Low-discrepancy_sequence
[4] Sergei Kucherenko and Yury Sytsko, "Application of deterministic
    low-discrepancy sequences in global optimization," Computational
    Optimization and Applications, vol. 30, p. 297-318, 2005

Author and Copyright: Geir Horn, 2018
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_NON_LINEAR_MLSL
#define OPTIMIZATION_NON_LINEAR_MLSL

#include <sstream>                            // For error reporting
#include <stdexcept>                          // For standard exceptions
#include <type_traits>                        // Meta programming

#include "../Variables.hpp"                   // Basic definitions

#include "NonLinear/Algorithms.hpp"           // Definition of the algorithms
#include "NonLinear/Objective.hpp"            // Objective function
#include "NonLinear/Optimizer.hpp"            // Optimizer interface
#include "NonLinear/Bounds.hpp"                    // Variable domain bounds

namespace Optimization::NonLinear
{

/*==============================================================================

 ML: Multi-level

==============================================================================*/
//
// The various variants of the multi-level algorithms differ in their
// requirements for a gradient or not, and these requirements extend to the
// sub algorithm used. It is therefore possible to derive them all from a
// common class.

template< Algorithm::ID PrimaryAlgorithm, Algorithm::ID SubsidiaryAlgorithm >
class MultiLevel
: virtual public NonLinear::Objective< SubsidiaryAlgorithm >,
  virtual public NonLinear::Bound,
	public NonLinear::OptimizerInterface
{
private:

	double         LocalObjectiveTolerance,
	               LocalVariableTolerance;

protected:

  virtual Algorithm::ID GetAlgorithm( void ) override
	{ return PrimaryAlgorithm;	}

	virtual Algorithm::ID GetSubsidiaryAlgorithm( void )
	{	return SubsidiaryAlgorithm; }

	inline void SetLocalObjectiveTolerance( double RelativeTolerance )
	{
		if ( RelativeTolerance > 0.0 )
			LocalObjectiveTolerance = RelativeTolerance;
	}

	inline double GetObjectiveTolerance( void )
	{ return LocalObjectiveTolerance; }

	inline void SetLocalVariableTolerance( double RelativeTolerance )
	{
		if ( RelativeTolerance > 0.0 )
			LocalVariableTolerance = RelativeTolerance;
	}

	inline double GetVariableTolerance( void )
	{ return LocalVariableTolerance; }

	// It is explicitly stated that the objective function, bounds, or
	// constraints set for the local search algorithm all will be ignored
	// and therefore thy will not be initialised by the create solver function.

	SolverPointer
	CreateSolver( Dimension NumberOfVariables, Objective::Goal Direction) override
	{
		SolverPointer TheSolver =
									OptimizerInterface::CreateSolver( NumberOfVariables,
																										Direction );

	  SetObjective( TheSolver, Direction );
		SetBounds( TheSolver );

		// Then the local solver is created

		SolverPointer LocalSolver =
		nlopt_create( static_cast< nlopt_algorithm >( SubsidiaryAlgorithm ),
									NumberOfVariables );

		if ( LocalSolver == nullptr )
		{
			std::ostringstream ErrorMessage;

			ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
			             << " Failed to create the local solver for a "
									 << "Multi Level method.";

		  throw std::runtime_error( ErrorMessage.str() );
		}

		// If the tolerances are given, they will be registered for the local
		// solver

		if ( LocalObjectiveTolerance > 0.0 )
			nlopt_set_ftol_rel( LocalSolver, LocalObjectiveTolerance );

		if ( LocalVariableTolerance > 0.0 )
			nlopt_set_xtol_rel( LocalSolver, LocalVariableTolerance );

		// Finally, the local solver is set for the global solver. This
		// creates a copy of the local solver object, and the local solver
		// must then be destroyed to free the memory.

		nlopt_set_local_optimizer( TheSolver, LocalSolver );
		nlopt_destroy( LocalSolver );

		// The initialised solver object is then returned.

		return TheSolver;
	}

	// The constructor may take the parameters for the problem

	MultiLevel( double ObjectiveTolerance = 0.0,
							double VariableTolerance = 0.0  )
	: Objective(), Bound(), OptimizerInterface(),
    LocalObjectiveTolerance( ObjectiveTolerance ),
    LocalVariableTolerance( VariableTolerance )
	{}

	// The destructor does basically nothing except ensures the right
	// destruction of the inherited classes.

public:

	virtual ~MultiLevel( void )
	{}
};

/*==============================================================================

 MLSL: Multi-level single linkage

==============================================================================*/
//
// The single linkage extension to this defines the number of starting points
// and sets this for the solver.

template< Algorithm::ID PrimaryAlgorithm, Algorithm::ID SubsidiaryAlgorithm >
class MultiLevelSingleLinkage
:  public MultiLevel< PrimaryAlgorithm,  SubsidiaryAlgorithm >
{
private:

	unsigned int   NumberOfStartingPoints;

	// A short hand for the base template is needed to call functions from the
	// multi-level base class requires full qualification.

	using ML = MultiLevel< PrimaryAlgorithm,  SubsidiaryAlgorithm >;

protected:

	inline void SetNumberOfStartingPoints( unsigned int n )
	{ NumberOfStartingPoints = n; }

  // The definition of the create solver function is extended by setting the
  // starting point for the solver. If it is not set, i.e. is zero, the
  // default value will be used.

 	SolverPointer	CreateSolver( Dimension NumberOfVariables,
 	                            Optimization::Objective::Goal Direction) override
	{

	  SolverPointer TheSolver = ML::CreateSolver( NumberOfVariables,  Direction );

	  if ( NumberOfStartingPoints > 0 )
			nlopt_set_population( TheSolver, NumberOfStartingPoints );

    return TheSolver;
	}

	// The constructor takes the problem parameters and forwards the tolerance
	// parameters to the multi-level class.

	MultiLevelSingleLinkage( unsigned int StartingPoints = 0,
						               double ObjectiveTolerance = 0.0,
							             double VariableTolerance = 0.0  )
  : ML( ObjectiveTolerance, VariableTolerance ),
    NumberOfStartingPoints( StartingPoints )
  {}

public:

	virtual ~MultiLevelSingleLinkage( void )
	{}
};


/*==============================================================================

 Non derivative Multi-Level Single-Linkage

==============================================================================*/

template< Algorithm::ID SubsidiaryAlgorithm >
class Optimizer< Algorithm::Global::MultiLevelSingleLinkage::NonDerivative,
      SubsidiaryAlgorithm,
      std::enable_if_t< !Algorithm::RequiresGradient( SubsidiaryAlgorithm ) > >
: public MultiLevelSingleLinkage<
				 Algorithm::Global::MultiLevelSingleLinkage::NonDerivative,
				 SubsidiaryAlgorithm >
{
private:

	using MLSL = MultiLevelSingleLinkage<
				       Algorithm::Global::MultiLevelSingleLinkage::NonDerivative,
				       SubsidiaryAlgorithm >;

protected:

	Optimizer( unsigned int StartingPoints = 0, double ObjectiveTolerance = 0.0,
	           double VariableTolerance = 0.0 )
  : MLSL( StartingPoints,  ObjectiveTolerance,  VariableTolerance )
  {}

public:

	virtual ~Optimizer( void )
	{}
};

/*==============================================================================

 Derivative Multi-Level Single-Linkage

==============================================================================*/
//
// If the derivative variant is used, it requires the gradients to be defined
// and in this case the sub-algorithm may or may not use the gradients, and
// it can therefore be any kind of algorithm. The only changes necessary is
// re-defining the sub-algorithm related functions.

template< Algorithm::ID SubsidiaryAlgorithm >
class Optimizer< Algorithm::Global::MultiLevelSingleLinkage::Derivative,
      SubsidiaryAlgorithm,
      std::enable_if_t< Algorithm::RequiresGradient( SubsidiaryAlgorithm ) > >
: public MultiLevelSingleLinkage<
				 Algorithm::Global::MultiLevelSingleLinkage::Derivative,
				 SubsidiaryAlgorithm >
{
private:

	using MLSL = MultiLevelSingleLinkage<
				       Algorithm::Global::MultiLevelSingleLinkage::Derivative,
				       SubsidiaryAlgorithm >;

protected:

	Optimizer( unsigned int StartingPoints = 0, double ObjectiveTolerance = 0.0,
	           double VariableTolerance = 0.0 )
  : MLSL( StartingPoints,  ObjectiveTolerance, VariableTolerance )
  {}

public:

	virtual ~Optimizer( void )
	{}
};

/*==============================================================================

 Non Derivative Low-discrepancy Multi-Level Single-Linkage

==============================================================================*/
//
// The only thing that makes this different from the normal MLSL non derivative
// algorithm is the top level algorithm definition.

template< Algorithm::ID SubsidiaryAlgorithm >
class Optimizer<
Algorithm::Global::MultiLevelSingleLinkage::LowDiscrepancySequence::NonDerivative,
SubsidiaryAlgorithm,
std::enable_if_t< !Algorithm::RequiresGradient( SubsidiaryAlgorithm ) > >
: public MultiLevelSingleLinkage<
Algorithm::Global::MultiLevelSingleLinkage::LowDiscrepancySequence::NonDerivative,
SubsidiaryAlgorithm >
{
private:

	using MLSL = MultiLevelSingleLinkage<
  Algorithm::Global::MultiLevelSingleLinkage::LowDiscrepancySequence::NonDerivative,
  SubsidiaryAlgorithm >;

protected:

	Optimizer( unsigned int StartingPoints = 0, double ObjectiveTolerance = 0.0,
	           double VariableTolerance = 0.0 )
  : MLSL( StartingPoints,  ObjectiveTolerance, VariableTolerance )
  {}

public:

	virtual ~Optimizer( void )
	{}
};

/*==============================================================================

 Derivative Low-discrepancy Multi-Level Single-Linkage

==============================================================================*/
//
// The variant requiring the gradient function is similar to the
// similar derivative based multi-level single linkage class.

template< Algorithm::ID SubsidiaryAlgorithm >
class Optimizer<
Algorithm::Global::MultiLevelSingleLinkage::LowDiscrepancySequence::Derivative,
SubsidiaryAlgorithm,
std::enable_if_t< Algorithm::RequiresGradient( SubsidiaryAlgorithm ) > >
:  public MultiLevelSingleLinkage<
Algorithm::Global::MultiLevelSingleLinkage::LowDiscrepancySequence::Derivative,
SubsidiaryAlgorithm >
{
private:

	using MLSL = MultiLevelSingleLinkage<
  Algorithm::Global::MultiLevelSingleLinkage::LowDiscrepancySequence::Derivative,
  SubsidiaryAlgorithm >;

protected:

	Optimizer( unsigned int StartingPoints = 0, double ObjectiveTolerance = 0.0,
	           double VariableTolerance = 0.0 )
  : MLSL( StartingPoints,  ObjectiveTolerance, VariableTolerance )
  {}

public:

	virtual ~Optimizer( void )
	{}
};

}      // Name space Optimization non-linear
#endif // OPTIMIZATION_NON_LINEAR_MLSL
