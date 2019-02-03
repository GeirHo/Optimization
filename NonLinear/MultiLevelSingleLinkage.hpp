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

#include "../Variables.hpp"                   // Basic definitions

#include "NonLinear/Algorithms.hpp"           // Definition of the algorithms
#include "NonLinear/Objective.hpp"            // Objective function
#include "NonLinear/Optimizer.hpp"            // Optimizer interface
#include "NonLinear/Bounds.hpp"               // Variable domain bounds

namespace Optimization::NonLinear
{
/*==============================================================================

 Non derivative Multi-Level Single-Linkage

==============================================================================*/

template<>
class Optimizer< Algorithm::Global::MultiLevelSingleLinkage::NonDerivative >
: virtual public NonLinear::Objective,
  virtual public NonLinear::Bound,
	public NonLinear::OptimizerInterface
{
private:
	
	Algorithm::ID  LocalSearchAlgorithm;
	unsigned int   NumberOfStartingPoints;
	double         LocalObjectiveTolerance,
	               LocalVariableTolerance;
	
protected:
	
	// There are small utility functions to set these parameters
	
	inline void SetLocalSearchAlgorithm( const Algorithm::ID TheAlgorithm  )
	{ 
	  if ( Algorithm::Requires_Gradient( TheAlgorithm ) )
		{
			std::ostringstream ErrorMessage;
			
			ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
									 << "The non-derivative variant of the multi-level single "
									 << "linkage algorithm must use a non-derivative local "
									 << "search algorithm";
									 
		  throw std::logic_error( ErrorMessage.str() );
		}
		else
			LocalSearchAlgorithm = TheAlgorithm; 
	}
	
  virtual Algorithm::ID GetAlgorithm( void ) override
	{ 
		return Algorithm::Global::MultiLevelSingleLinkage::NonDerivative;
	}

	virtual Algorithm::ID GetLocalSearchAlgorithm( void )
	{
		return LocalSearchAlgorithm;
	}
	
	inline void SetNumberOfStartingPoints( unsigned int n )
	{ NumberOfStartingPoints = n; }
	
	inline void SetLocalObjectiveTolerance( double RelativeTolerance )
	{ 
		if ( RelativeTolerance > 0.0 ) 
			LocalObjectiveTolerance = RelativeTolerance;
	}
	
	inline void SetLocalVariableTolerance( double RelativeTolerance )
	{
		if ( RelativeTolerance > 0.0 )
			LocalVariableTolerance = RelativeTolerance;
	}
		
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
		
		// Testing and setting the initial population size
		
		if ( NumberOfStartingPoints > 0 )
			nlopt_set_population( TheSolver, NumberOfStartingPoints );
		
		// Then the local solver is created
		
		SolverPointer LocalSolver = 
		nlopt_create( static_cast< nlopt_algorithm >( GetLocalSearchAlgorithm() ),
									NumberOfVariables );
		
		if ( LocalSolver == nullptr )
		{
			std::ostringstream ErrorMessage;
			
			ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
			             << " Failed to create the local solver for the "
									 << "Multi Level Single Linkage method.";
									 
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

	// The constructor requires the algorithm of the sub-solver
	
	Optimizer( Algorithm::ID SubAlgorithm )
	: Objective(), Bound(), OptimizerInterface()
	{
		SetLocalSearchAlgorithm( SubAlgorithm );;
		NumberOfStartingPoints  = 0;
		LocalObjectiveTolerance = 0.0;
		LocalVariableTolerance  = 0.0;
	}	
	
	Optimizer( void ) = delete;
	
	// The destructor does basically nothing except ensures the right 
	// destruction of the inherited classes.

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

template<>
class Optimizer< Algorithm::Global::MultiLevelSingleLinkage::Derivative >
: virtual public NonLinear::ObjectiveGradient,
  virtual public NonLinear::Bound,
	public Optimizer< Algorithm::Global::MultiLevelSingleLinkage::NonDerivative >
{
private:
	
	// Since the algorithm definition is substantial in length, it is better 
	// to define it as a known abbreviation in this class.
	
	using MLSL = 
	      Optimizer< Algorithm::Global::MultiLevelSingleLinkage::NonDerivative >;

	// Since the MLSL base class will not allow a gradient based algorithm
  // to be used, the local algorithm must be stored again here to be 
  // returned by the corresponding read-only function.
														 
	Algorithm::ID  LocalSearchAlgorithm;
	
protected:

	inline void SetLocalSearchAlgorithm( Algorithm::ID TheAlgorithm  )
	{ 
		LocalSearchAlgorithm = TheAlgorithm; 
	}

  virtual Algorithm::ID GetAlgorithm( void ) override
	{ 
		return Algorithm::Global::MultiLevelSingleLinkage::Derivative;
	}

	virtual Algorithm::ID GetLocalSearchAlgorithm( void ) final
	{
		return LocalSearchAlgorithm;
	}
	
	// The constructor takes the sub-algorithm and passes this on to the 
	// MLSL base class if the algorithm is gradient free. Most likely it is 
	// not and the algorithm is simply passed as the algorithm enumerator. 
	// This will do no harm as the algorithm will never be used to initialize 
	// any solver. The local search algorithm for this version may or may not 
	// use the provided gradient, and there is no reason to test its value
	// before assigning it as the local search algorithm.
		
	Optimizer( Algorithm::ID SubAlgorithm )
	: ObjectiveGradient(), Bound(), 
	  MLSL( Algorithm::Requires_Gradient( SubAlgorithm ) ? 
				  Algorithm::ID::NUM_ALGORITHMS : SubAlgorithm ),
		LocalSearchAlgorithm( SubAlgorithm )
	{}
	
	Optimizer( void )= delete;
	
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

template<>
class Optimizer< 
Algorithm::Global::MultiLevelSingleLinkage::LowDiscrepancySequence::NonDerivative >
: virtual public NonLinear::Objective,
  virtual public NonLinear::Bound,
	public Optimizer< Algorithm::Global::MultiLevelSingleLinkage::NonDerivative >
{
private:
	
	using MLSL = 
	      Optimizer< Algorithm::Global::MultiLevelSingleLinkage::NonDerivative >;
														
protected:
	
	virtual Algorithm::ID GetAlgorithm( void ) final
	{
		return 
		Algorithm::Global::MultiLevelSingleLinkage::LowDiscrepancySequence::NonDerivative;
	}
	
	Optimizer( Algorithm::ID SubAlgorithm )
	: Objective(), Bound(), MLSL( SubAlgorithm )
	{}
	
	Optimizer( void ) = delete;
	
public:
	
	virtual ~Optimizer( void )
	{}
};

/*==============================================================================

 Derivative Low-discrepancy Multi-Level Single-Linkage

==============================================================================*/
//
// The variant requiring the gradient function is similarly derived from the 
// similar derivative based multi-level single linkage class.

template<>
class Optimizer< 
Algorithm::Global::MultiLevelSingleLinkage::LowDiscrepancySequence::Derivative >
: virtual public NonLinear::ObjectiveGradient,
  virtual public NonLinear::Bound,
	public Optimizer< Algorithm::Global::MultiLevelSingleLinkage::Derivative >
{
private:
	
	using MLSL = 
	      Optimizer< Algorithm::Global::MultiLevelSingleLinkage::Derivative >;
														
protected:
	
	virtual Algorithm::ID GetAlgorithm( void ) final
	{
		return 
		Algorithm::Global::MultiLevelSingleLinkage::LowDiscrepancySequence::Derivative;
	}

	Optimizer( Algorithm::ID SubAlgorithm )
	: Objective(), Bound(), MLSL( SubAlgorithm )
	{}
	
	Optimizer( void ) = delete;
	
public:
	
	virtual ~Optimizer( void )
	{}
};

	
}      // Name space Optimization non-linear
#endif // OPTIMIZATION_NON_LINEAR_MLSL
