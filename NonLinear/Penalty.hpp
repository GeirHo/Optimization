/*==============================================================================
Penalty

This file implements the specialisations for some algorithms that augment the
objective function with a penalty for solutions outside the feasible region.
The Augmented Lagrangian method [1,2] is the most general approach, and the
subsidiary solver can be gradient based or non-gradient based. Both global or
local optimisers can be used, although Johnson warns that it can be difficult
to set a good stopping criterion for a global optimizer. Note that several
variants exists of the specialisation and the compiler may issue odd errors
if the legal combinations are not fulfilled. For instance, if one asks for a
subsidiary local optimisation algorithm but specifies a global penalty method.

There are also two other solvers using a penalty when moving outside the
feasible region: The conservative convex separable approximation and the
method of moving asymptotes [3]. The first method allows a user defined
approximation to the Hessian matrix (Gradient Matrix).

References:

[1] Andrew R. Conn, Nicholas I. M. Gould, and Philippe L. Toint, "A globally
    convergent augmented Lagrangian algorithm for optimization with general
    constraints and simple bounds," SIAM J. Numer. Anal. vol. 28, no. 2,
		pp. 545-572, 1991
[2] E. G. Birgin and J. M. Martínez, "Improving ultimate convergence of an
    augmented Lagrangian method," Optimization Methods and Software vol. 23,
		no. 2, p. 177-195, 2008.
[3] Krister Svanberg, "A class of globally convergent optimization methods
    based on conservative convex separable approximations," SIAM J. Optim.
    Vol. 12, No. 2, pp. 555-573, 2002

Author and Copyright: Geir Horn, 2018
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_NON_LINEAR_PENALTY
#define OPTIMIZATION_NON_LINEAR_PENALTY

#include <type_traits>                        // For meta-programming

#include "../Variables.hpp"                   // Basic definitions

#include "NonLinear/Algorithms.hpp"              // Definition of the algorithms
#include "NonLinear/Objective.hpp"               // Objective function
#include "NonLinear/Optimizer.hpp"               // Optimizer interface
#include "NonLinear/Bounds.hpp"                  // Variable domain bounds
#include "NonLinear/Constraints.hpp"             // Constraint functions
#include "NonLinear/MultiLevelSingleLinkage.hpp" // Base class

namespace Optimization::NonLinear
{
/*==============================================================================

 Augmented Lagrangian method

==============================================================================*/
//
// There are several variants of this algorithm that must be supported since it
// can wrap almost any of the other algorithms. The different variants all share
// the same way to create the solver and set the sub-solver, and therefore a
// common base class can be used to avoid duplicating code.
//
// However, the tasks of defining the primary solver and a subsidiary solver and
// bind the solvers are identical to the functionality of any multi-level solver
// the the multi-level base class is reused. It is defined in the header
// for the Multi Level Single Linkage algorithm.
//
// It should be noted that the Lagrangian method always allows both inequality
// and equality constraints. Depending on the algorithm, the constraints are
// either directly included into the penalty, or the inequality constraints
// could be passed on to the subsidiary algorithm. However, this functionality
// decision is taken by the NLOpt library and it does not have to be explicitly
// formulated. The constraints just have to be defined when the solver is
// created.
//
// Finally, the template will only match if the subsidiary algorithm is not
// itself a multi-level algorithm.

template< Algorithm::ID PrimaryAlgorithm, Algorithm::ID SubsidiaryAlgorithm,
          typename = std::enable_if_t<
						!Algorithm::RequiresSubsidiary( SubsidiaryAlgorithm ) > >
class LagrangianSolver
: virtual public NonLinear::InEqConstraints< PrimaryAlgorithm >,
  virtual public NonLinear::EqConstraints< PrimaryAlgorithm >,
  public MultiLevel< PrimaryAlgorithm, SubsidiaryAlgorithm >
{
private:

  // Defining shorthand notation for the base classes

  using InEq       = NonLinear::InEqConstraints< PrimaryAlgorithm >;
	using Eq         = NonLinear::EqConstraints< PrimaryAlgorithm >;
	using MultiLevel = MultiLevel< PrimaryAlgorithm, SubsidiaryAlgorithm >;

protected:

  // The create solver function adds the constraints if they have been defined
  // prior to calling the create solver function.

  SolverPointer	CreateSolver( Dimension NumberOfVariables,
								              Optimization::Objective::Goal Direction) override
	{
		SolverPointer TheSolver =
	                MultiLevel::CreateSolver( NumberOfVariables, Direction );

    if ( InEq::NumberOfInEqConstraints() > 0 )
      SetInEqConstraints( TheSolver, MultiLevel::GetVariableTolerance() );

    if ( Eq::NumberOfEqConstraints() > 0 )
      SetEqConstraints( TheSolver, MultiLevel::GetVariableTolerance() );

    return TheSolver;
	}

	// The constructor takes the two tolerances and forwards them to the multi-
  // level base class

	LagrangianSolver( double ObjectiveTolerance = 0.0,
                    double VariableTolerance  = 0.0 )
	: InEq(),  Eq(),	MultiLevel( ObjectiveTolerance, VariableTolerance )
	{}

public:

	virtual ~LagrangianSolver( void )
	{}
};

// -----------------------------------------------------------------------------
// All constraints in the penalty function
// -----------------------------------------------------------------------------
//
// The global Lagrangian specialization requires the secondary algorithm to be
// a global algorithm, and puts no other constraints on the secondary algorithm.
// Ideally, this should support setting more elaborate stopping criteria for
// the local solver, but it is not evident what are the best stopping criteria
// and so this is left for user extensions.

template< Algorithm::ID SubsidiaryAlgorithm >
class Optimizer< Algorithm::Global::Penalty::AllConstraints,
      SubsidiaryAlgorithm,
      std::enable_if_t< Algorithm::IsGlobal( SubsidiaryAlgorithm ) > >
: public LagrangianSolver< Algorithm::Global::Penalty::AllConstraints,
                           SubsidiaryAlgorithm >
{
protected:

  // The constructor takes the tolerances and forwards these to the Lagrangian
  // solver's constructor

  Optimizer( double ObjectiveTolerance = 0.0, double VariableTolerance  = 0.0 )
	: LagrangianSolver< Algorithm::Global::Penalty::AllConstraints,
	    SubsidiaryAlgorithm >( ObjectiveTolerance,  VariableTolerance )
	{}

public:

	virtual ~Optimizer( void )
	{}
};

// Similarly, the local solver requires a local algorithm. Otherwise it is
// identical to the global version.

template< Algorithm::ID SubsidiaryAlgorithm >
class Optimizer< Algorithm::Local::Penalty::AllConstraints,
      SubsidiaryAlgorithm,
      std::enable_if_t< Algorithm::IsLocal( SubsidiaryAlgorithm ) > >
: public LagrangianSolver< Algorithm::Local::Penalty::AllConstraints,
                           SubsidiaryAlgorithm >
{
protected:

  // The constructor takes the tolerances and forwards these to the Lagrangian
  // solver's constructor

  Optimizer( double ObjectiveTolerance = 0.0, double VariableTolerance  = 0.0 )
	: LagrangianSolver< Algorithm::Local::Penalty::AllConstraints,
	    SubsidiaryAlgorithm >( ObjectiveTolerance,  VariableTolerance )
	{}

public:

	virtual ~Optimizer( void )
	{}
};

// -----------------------------------------------------------------------------
// Equality constraints in the penalty function
// -----------------------------------------------------------------------------
//
// There is an algorithmic variant that assumes that the secondary algorithm is
// capable of handling inequality constraints.Otherwise the implementation is
// similar to the above implementations.

template< Algorithm::ID SubsidiaryAlgorithm >
class Optimizer< Algorithm::Global::Penalty::EqualityConstraints,
      SubsidiaryAlgorithm,
      std::enable_if_t<
				Algorithm::IsGlobal( SubsidiaryAlgorithm ) &&
				Algorithm::SupportsInequalityConstraints( SubsidiaryAlgorithm ) > >
: public LagrangianSolver< Algorithm::Global::Penalty::EqualityConstraints,
                           SubsidiaryAlgorithm >
{
protected:

  // The constructor takes the tolerances and forwards these to the Lagrangian
  // solver's constructor

  Optimizer( double ObjectiveTolerance = 0.0, double VariableTolerance  = 0.0 )
	: LagrangianSolver< Algorithm::Global::Penalty::EqualityConstraints,
	    SubsidiaryAlgorithm >( ObjectiveTolerance,  VariableTolerance )
	{}

public:

	virtual ~Optimizer( void )
	{}
};

// The local variant is the same except that the subsidiary algorithm is
// required to be local too.

template< Algorithm::ID SubsidiaryAlgorithm >
class Optimizer< Algorithm::Local::Penalty::EqualityConstraints,
      SubsidiaryAlgorithm,
      std::enable_if_t<
				Algorithm::IsLocal( SubsidiaryAlgorithm ) &&
				Algorithm::SupportsInequalityConstraints( SubsidiaryAlgorithm ) > >
: public LagrangianSolver< Algorithm::Local::Penalty::EqualityConstraints,
                           SubsidiaryAlgorithm >
{
protected:

  // The constructor takes the tolerances and forwards these to the Lagrangian
  // solver's constructor

  Optimizer( double ObjectiveTolerance = 0.0, double VariableTolerance  = 0.0 )
	: LagrangianSolver< Algorithm::Local::Penalty::EqualityConstraints,
	    SubsidiaryAlgorithm >( ObjectiveTolerance,  VariableTolerance )
	{}

public:

	virtual ~Optimizer( void )
	{}
};

/*==============================================================================

 CCSA: Conservative convex separable approximation

==============================================================================*/
//
// It should be noted that this function is based on the objective gradient
// class and also requires the gradient of the inequality constraints.

template<>
class Optimizer< Algorithm::Local::Penalty::ConvexSeparable >
: virtual public
  NonLinear::Objective< Algorithm::Local::Penalty::ConvexSeparable >,
  virtual public NonLinear::Bound,
  virtual public
  NonLinear::InEqConstraints< Algorithm::Local::Penalty::ConvexSeparable >,
	public NonLinear::OptimizerInterface
{
private:

  // The parameter for the constraint tolerance must again be stored.

  double Tolerance;

protected:

  // Define the algorithm used by the solver of this optimizer

  virtual Algorithm::ID GetAlgorithm( void ) override
	{ return Algorithm::Local::Penalty::ConvexSeparable; }

  // The function to create the solver will also initialise the bounds and
  // the inequality constraints for the problem

	SolverPointer CreateSolver( Dimension NumberOfVariables,
														  Objective::Goal Direction) final
	{
		SolverPointer TheSolver =
									OptimizerInterface::CreateSolver( NumberOfVariables,
																										Direction);

    SetObjective( TheSolver, Direction );
		SetBounds( TheSolver );

    // Since the constraints are optional they may not be given and they are
    // set only if they have been provided.

    if ( NumberOfInEqConstraints() > 0 )
      SetInEqConstraints( TheSolver, Tolerance );

    return TheSolver;
	}

  // The constructor is protected to ensure that it only will be called by
	// derived classes defining the objective function and the constraints.
	// Its main purpose is to initialise the base classes. Note that the
	// virtual base classes must be initialised before the interface class.

	Optimizer( double ConstraintTolerance )
	: ObjectiveGradient(), Bound(), InEqConstraints(), OptimizerInterface(),
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

 MMA: Method of Moving Asymptotes

==============================================================================*/
//
// The MMA is an evolution of the CCSA algorithm and it is therefore based
// on the CSSA implementation modifying just the function to return the
// right algorithm type.

template<>
class Optimizer< Algorithm::Local::Penalty::MovingAsymptotes >
: public Optimizer< Algorithm::Local::Penalty::ConvexSeparable >
{
protected:

  // Define the algorithm used by the solver of this optimizer

  virtual Algorithm::ID GetAlgorithm( void ) override
	{ return Algorithm::Local::Penalty::MovingAsymptotes; }

	Optimizer( double ConstraintTolerance )
	: Optimizer< Algorithm::Local::Penalty::ConvexSeparable >(
		           ConstraintTolerance )
	{}

  Optimizer( void ) = delete;

public:

  // The destructor is virtual to ensure that the sub-classes are properly
  // destroyed although it does nothing in particular for this class.

  virtual ~Optimizer( void )
  {}
};

}      // end name space Non-linear Optimization
#endif // OPTIMIZATION_NON_LINEAR_PENALTY
