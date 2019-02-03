/*==============================================================================
LOCAL APPROXIMATIONS

This file implements the specialisations for the algorithms that uses some 
kind of interpolation of the objective function to identify the optimum. They
all support bounds on the search space, but only the linear approximation [1,2]
supports general constraints.

The quadratic approximation algorithm solves quadratic subproblems in a 
spherical trust region via a truncated conjugate-gradient algorithm [3]. 
This algorithm supports bound constraints, but because it constructs a 
quadratic approximation of the objective, it may perform poorly for objective 
functions that are not twice-differentiable.

The third algorithm [4] is an extension of the quadratic approximation that 
supports unequal initial-step sizes in the different parameters by the simple 
expedient of internally rescaling the parameters proportional to the initial 
steps, which is important when different parameters have very different scales.
Otherwise this is similar to the quadratic approximaton, and it should therfore
be prefered.

References:

[1] M. J. D. Powell: "A direct search optimization method that models the 
    objective and constraint functions by linear interpolation," in Advances 
    in Optimization and Numerical Analysis, eds. S. Gomez and J.-P. Hennart 
    p. 51-67, Kluwer Academic: Dordrecht, 1994.
[2] M. J. D. Powell, "Direct search algorithms for optimization calculations," 
    Acta Numerica, Vol. 7, pp. 287-336, 1998
[3] M. J. D. Powell, "The NEWUOA software for unconstrained optimization 
    without derivatives," Proc. 40th Workshop on Large Scale Nonlinear 
    Optimization, Erice, Italy, 2004
[4] M. J. D. Powell, "The BOBYQA algorithm for bound constrained optimization 
    without derivatives," Department of Applied Mathematics and Theoretical 
    Physics, Cambridge England, technical report NA2009/06, 2009

Author and Copyright: Geir Horn, 2019
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_NON_LINEAR_LOCAL_APPROXIMATION
#define OPTIMIZATION_NON_LINEAR_LOCAL_APPROXIMATION

#include "../Variables.hpp"                   // Basic definitions

#include "NonLinear/Algorithms.hpp"           // Definition of the algorithms
#include "NonLinear/Objective.hpp"            // Objective function
#include "NonLinear/Optimizer.hpp"            // Optimizer interface
#include "NonLinear/Bounds.hpp"               // Variable domain bounds
#include "NonLinear/Constraints.hpp"          // Constraint functions


namespace Optimization::NonLinear
{
/*==============================================================================

 COBYLA: Constrained Optimization by Linear Approximations

==============================================================================*/
//
// The algorithm works by constructing successive linear approximations of 
// the objective function and constraints via a simplex of n+1 points in n 
// dimensions, and optimizes these approximations in a trust region at each 
// step.

template<>
class Optimizer< Algorithm::Local::NonDerivative::LinearApproximation >
: virtual public NonLinear::Objective,
  virtual public NonLinear::Bound,
  virtual public NonLinear::InEqConstraints,
  virtual public NonLinear::EqConstraints,
	public NonLinear::OptimizerInterface
{
private:

  // There is a parameter defining the tolerance usd for the inequality 
  // and the equality constraints as completely satisfying the constraints 
  // may take much longer and add little to the solution quality.

  double Tolerance;

protected:

  // Define the algorithm used by the solver of this optimizer
	
  virtual Algorithm::ID GetAlgorithm( void ) override
	{ 
		return Algorithm::Local::NonDerivative::LinearApproximation;	
	}
	
  // The function to create the solver will also initialise the bounds and 
  // the constraints for the problem 
	
	SolverPointer CreateSolver( Dimension NumberOfVariables, 
														  Objective::Goal Direction) final
	{
		SolverPointer TheSolver = 
									OptimizerInterface::CreateSolver( NumberOfVariables, 
																										Direction);
		
    SetObjective( TheSolver, Direction );
		SetBounds( TheSolver );
    
    // Since the constraints are optional they may not be given and they are
    // set only if they are provided.
    
    if ( NumberOfInEqConstraints() > 0 )
      InEqConstraints::SetInEqConstraints( TheSolver, Tolerance );
      
    if ( NumberOfEqConstraints() > 0 )
      EqConstraints::SetEqConstraints( TheSolver, Tolerance );
    
    return TheSolver;
	}

  // The constructor is protected to ensure that it only will be called by 
	// derived classes defining the objective function and the bound function.
	// Its main purpose is to initialise the base classes. Note that the 
	// virtual base classes must be initialised before the interface class.
	
	Optimizer( double ConstraintTolerance )
	: Objective(), Bound(), InEqConstraints(), EqConstraints(),
    OptimizerInterface(),
	  Tolerance( ConstraintTolerance )
	{}
  
  Optimizer( void ) = delete;
  
public:

  // The destructor is virtual to ensure that the sub-classes are properly 
  // destroyed although it does nothing in particular for this class.
  
  virtual ~Optimizer( void )
  {}
};

/*==============================================================================

 NEWUOA: Bound Optimization by Quadratic Approximations

==============================================================================*/
//
// This is Steven G. Johnson's modification of the original NEWUOA algorithm 
// by useing the MMA algorithm for the quadratic subproblems and solve them 
// with both bound constraints and a spherical trust region. It should be 
// noted that the algoritm requires two or more dimensions and it does not 
// handle one-dimensional optimization problems.

template<>
class Optimizer< Algorithm::Local::NonDerivative::QuadraticApproximation >
: virtual public NonLinear::Objective,
  virtual public NonLinear::Bound,
	public NonLinear::OptimizerInterface
{
protected:

  // Define the algorithm used by the solver of this optimizer
	
  virtual Algorithm::ID GetAlgorithm( void ) override
	{ 
		return Algorithm::Local::NonDerivative::QuadraticApproximation;	
	}
	
  // The function to create the solver will also initialise the bounds and 
  // the constraints for the problem 
	
	SolverPointer CreateSolver( Dimension NumberOfVariables, 
														  Objective::Goal Direction) final
	{
		SolverPointer TheSolver = 
									OptimizerInterface::CreateSolver( NumberOfVariables, 
																										Direction);
		
	  SetObjective( TheSolver, Direction );
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
  
public:
  
  virtual ~Optimizer( void )
  {}
};

/*==============================================================================

 BOBYQA: Bound Optimization by Quadratic Approximations

==============================================================================*/
//
// The algorithm is an enhancement of the NEWUOA algorithm and supports unequal
// initial-step sizes in the different parameters by the simple expedient of 
// internally rescaling the parameters proportional to the initial steps, 
// which is important when different parameters have very different scales.
// It should therefore be preferred over the NEWUOA algorithm for bounded 
// problems, but it may also suffer from poor performance if the objective 
// function is not twice differentiable.
// 
// The interface is therefore directly based on the NEWUOA interface, and it 
// shares the function to create the solver.

template<>
class Optimizer< Algorithm::Local::NonDerivative::RescalingApproximation >
: virtual public NonLinear::Objective,
  virtual public NonLinear::Bound,
	public Optimizer< Algorithm::Local::NonDerivative::QuadraticApproximation >
{
protected:

  // Define the algorithm used by the solver of this optimizer
	
  virtual Algorithm::ID GetAlgorithm( void ) override
	{ 
		return Algorithm::Local::NonDerivative::RescalingApproximation;	
	}
  
  // The constructor must initialise the virtual base classes, but otherwise 
  // it does nothing.
  
  Optimizer( void )
  : Objective(), Bound(), 
    Optimizer< Algorithm::Local::NonDerivative::QuadraticApproximation >()
  {}
  
public:

  virtual ~Optimizer( void )
  {}  
};

}       // end name space Optimization non-linear
#endif  // OPTIMIZATION_NON_LINEAR_LOCAL_APPROXIMATION
