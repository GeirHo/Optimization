/*==============================================================================
Constraints

Constraints are the most important aspect of any optimisation problem since 
inequality constraints confine the search space and equality constraints 
reduce the problem dimension. 

Fundamentally a constraint is a scalar function of a vector of variable values.
Furthermore, some optimisation algorithms will also need to evaluate the 
constraint gradient.The gradient is the partial derivative of the constraint 
function with respect to each of the variables. In other words, if there are 
n variables, then the gradient is a vector of n values - for each constraint.
Hence, the constraints are gradients represented as an Armadillo [1] matrix.
If C[j] is constraint j the column j consists of the partial derivatives of 
this with respect to each variable value x[i], in other words dC[j]/dx[i]. The 
reason for having the columns representing the constraints is because Armadillo 
stores the matrix in column order.

The constraints class is therefore just an overall framework, and the derived 
classes for vector constraints and individual constraints defines the functions
that evaluates the constraints.

A variable assignment is feasible if it satisfies all constraints.

References:

[1] Conrad Sanderson and Ryan Curtin: Armadillo: a template-based C++ library
		for linear algebra. Journal of Open Source Software, Vol. 1, pp. 26, 2016. 
		http://arma.sourceforge.net/

Author and Copyright: Geir Horn, 2018
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_CONSTRAINTS
#define OPTIMIZATION_CONSTRAINTS

#include <vector>											        // For variables and values
#include <functional>                         // Constraint functions
#include <sstream>                            // Formatted error messages
#include <stdexcept>                          // Standard exceptions
#include <algorithm>                          // Iterator based algorithms
#include <armadillo>                          // Matrix library

#include "Variables.hpp"

namespace Optimization
{
/*==============================================================================

 Constraints

==============================================================================*/
//
// A constraint is fundamentally a function taking the variable values and 
// returning a scalar value of the constraint to be tested against a threshold 
// value. It is therefore defined in terms of the standard function.

using Constraint = 
			std::function< VariableType( const Variables & VariableValues ) >;

// The values of the constraints are assumed to be of the same type as the 
// variables, hence the constraint values is a vector of that type

using ConstraintValues = std::vector< VariableType >;

// The basic constraints class maintains a vector of constraints. There is a 
// function to add individual constraints and virtual functions to compute the 
// constraints and the gradient of the constraints. 

class Constraints
{
public:
	
	// In many situations it could be useful to have a reference to a particular 
	// constraint function. However, a plain pointer cannot be used as it could 
	// be invalidated in special situations where the constraint class is 
	// removed or if the referenced constraint is invalidated. A handler must 
	// therefore be used, and the standard shared pointer implements this in 
	// a good way.
	
	using Reference = std::shared_ptr< Constraint >;
	

private:
	
	// The consequence of allowing use code have a reference to a constraint 
	// is that the the constraint handlers are stored when the constraint is 
	// created
	
	std::vector< Reference > ConstraintFunctions;

protected:
	
	// The function returning the number of constraints returns the number of 
	// functions in the constraint functions vector. 
	
	inline Dimension NumberOfConstraints( void )
	{ return ConstraintFunctions.size(); }
		
	// The constraints can be added to the end of the vector of functions,
	// and a reference to the stored reference is returned by the emplace 
	// back function. This is used to create the retuned reference as 
	// a copy.
	
	inline Reference Add( const Constraint & TheConstraintFunction )
	{
		Reference NewConstraintHandler = 
							std::make_shared< Constraint >( TheConstraintFunction );
							
		ConstraintFunctions.emplace_back( NewConstraintHandler );
		
		return NewConstraintHandler;
	}

	// VALUES
	//	
	// Individual constraints can be evaluated by a constraint index and 
	// the variable values
	
	inline VariableType Value( Dimension i, const Variables & VariableValues )
	{
		if ( i < ConstraintFunctions.size() )
			return (*ConstraintFunctions[ i ])( VariableValues );
		else
		{
			std::ostringstream ErrorMessage;
			
			ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
			             << "Value of constraint " << i << " requested which "
									 << "is outside of the legal range [0,"
									 << ConstraintFunctions.size() << ")";
									 
		  throw std::invalid_argument( ErrorMessage.str() );
		}
	}
	
	// The same can also be obtained by the functor operator
	
	inline
	VariableType operator() ( Dimension i, const Variables & VariableValues )
	{
		return Value( i, VariableValues );
	}
	
	// The vector function returns a vector of values by evaluating each 
	// constraint for the given argument. It will throw an exception if no 
	// constraint functions have been defined. 
	
	inline ConstraintValues Value( const Variables & VariableValues )
	{ 
		if ( ConstraintFunctions.empty() )
		{
			std::ostringstream ErrorMessage;
			
			ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
			             << "Constraint vector evaluation cannot be done with "
									 << "no constraint functions defined";
									 
		  throw std::logic_error( ErrorMessage.str() );
		}
		else
		{
			ConstraintValues Values;
			
			for ( const Reference & ConstraintHandler : ConstraintFunctions )
				Values.push_back( (*ConstraintHandler)( VariableValues ) );
			
			return Values;
		}
	}

	// The constructor simply initialises the function vector.

public:
	
	Constraints( void ) : ConstraintFunctions()
	{}
};

// The gradient functions follows the same pattern. A gradient function will 
// however return a vector which contains the partial derivatives of the 
// constraint function with respect to each of the variables. It is also a 
// requirement that the gradient function is given at the same time as the 
// constraint function.

using GradientMatrix = arma::Mat< VariableType >;

using ConstraintGradient = std::function< 
      GradientVector( const Variables & VariableValues ) >;
			
class GradientConstraints : private Constraints
{
public:
  
  // The declaration of the reference to a constraint function is reused
  
  using ConstraintReference = Constraints::Reference;

	// Similar to the constraints, it should be possible to reference the 
	// gradient function through a smart pointer to ensure that the function 
	// remains available as long as it is referenced.
	
	using GradientReference = std::shared_ptr< ConstraintGradient >;
				
	// When adding a constraint it is mandatory to provide also the gradient 
	// function. Hence, the function adding the constraint must return both 
	// references, and a reference class is needed for this
				
	class ConstraintGradientReferences
	{
	public:
		
		const ConstraintReference TheConstraintReference;
		const GradientReference   TheGradientReference;
		
		ConstraintGradientReferences( 
      const ConstraintReference & TheConstraintFunction, 
      const GradientReference   & TheGradientFunction )
		: TheConstraintReference( TheConstraintFunction ),
		  TheGradientReference  ( TheGradientFunction   )
		{}
		
		ConstraintGradientReferences( const ConstraintGradientReferences & Other )
		: TheConstraintReference( Other.TheConstraintReference ),
		  TheGradientReference  ( Other.TheGradientReference   )
		{}
		
		ConstraintGradientReferences( void ) = delete;
	};

private:
	
	// The consequence is also in this case that the handlers are stored to 
	// represent the gradient functions.
	
	std::vector< GradientReference > GradientFunctions;
	
	
protected:
	  
	// The constraints can be evaluated using the value functions of the base 
	// class, and the number of constraints.
	
	using Constraints::Value;
	using Constraints::NumberOfConstraints;
	
	// The function to add the constraint function also requires the gradient 
	// function when this class is used, and it returns a reference containing 
	// both function references.
	
	inline ConstraintGradientReferences Add( 
         const Constraint & TheConstraintFunction, 
         const ConstraintGradient & TheGradientFunction )
	{
		GradientReference NewGradientHandler = 
								std::make_shared< ConstraintGradient >( TheGradientFunction );
								
		GradientFunctions.emplace_back( NewGradientHandler );
								
		return 
    ConstraintGradientReferences( Constraints::Add( TheConstraintFunction ),
											            NewGradientHandler );
	}
	
	// The gradient vector of a particular indexed constraint can be obtained by 
	// the gradient function.if it is called with only an index as argument. A 
	// test is performed to verify that the computed gradient vector has the same 
	// dimension as the variable vector. If not, a logic error is thrown.
	
	inline GradientVector Gradient( Dimension i, 
                                  const Variables & VariableValues )
	{ 
		if ( i < GradientFunctions.size() )
		{
			GradientVector GradientValues( 
																	(*GradientFunctions[ i ])( VariableValues ) );
			
			if ( GradientValues.size() == VariableValues.size() )
				return GradientValues;
			else
		  {
				std::ostringstream ErrorMessage;
				
				ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
				             << "The gradient function " << i << " returned a gradient "
										 << "vector with " << GradientValues.size() 
										 << " elements while it should have the same length as the"
										 << " variable vector (" << VariableValues.size() << ")";
										 
			  throw std::logic_error( ErrorMessage.str() );
			}
		}
		else
		{
			std::ostringstream ErrorMessage;
			
			ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
			             << "Value of gradient vector " << i << " requested which "
									 << "is outside of the legal range [0,"
									 << GradientFunctions.size() << ")";
									 
		  throw std::invalid_argument( ErrorMessage.str() );
		}
	}
	
	// In many situations it is better to work with the full gradient matrix
	// which will have one row per variable and one column per constraint. 
	
	inline GradientMatrix Gradient( const Variables & VariableValues )
	{
		Dimension ConstraintCount = GradientFunctions.size();
							
		GradientMatrix Gradients( VariableValues.size(), ConstraintCount, 
															arma::fill::zeros );
		
		// The constraint gradients are computed one by one and copied to the 
		// corresponding column of the matrix.
		
		for ( Dimension TheConstraint = 0; TheConstraint < ConstraintCount; 
				  TheConstraint++ )
	  {
		  GradientVector GradientValues( 
									   Gradient( TheConstraint, VariableValues ) );
	
		  std::copy( GradientValues.begin(), GradientValues.end(), 
								 Gradients.begin_col( TheConstraint ) );
		}
		
		return Gradients;
	}
	
	// The constructor simply initialises the gradient function vector
	
public:
	
	GradientConstraints( void )
	: Constraints(), GradientFunctions()
	{}
	
};

/*==============================================================================

 Inequality constraints

==============================================================================*/
//
// In most problems it is useful to distinguish between inequality constraints 
// and equality constraints because the former constraints the search space and 
// the latter reduces the dimensionality of the problem. For algorithms 
// supporting both types of constraints it would mean that the constraint class
// should be inherited twice, and this is clearly not possible. 
//
// The solution is to define separate classes for the two types of constraints
// based on the generic ones and allowing qualified addition of constraints, 
// like InEqConstraint::Add() and EqConstraint::Add() which would correctly 
// call the right add function.

class InEqConstraints : private Constraints
{
protected:
	
  // The declaration of the reference to a constraint function is reused
  
  using Constraints::Reference;
  
	// Adding constraints individually can be handled by the standard add function.
	
	using Constraints::Add;
	
	// The number of constraints are re-defined to be able to distinguish between 
	// the type of constraint. This is virtual for the situations where the 
	// constraints are not individually defined.
	
	virtual Dimension NumberOfInEqConstraints( void )
	{ return Constraints::NumberOfConstraints(); }
	
	// In the same way one may want to re-define the way the vector of the 
	// constraint values is computed and there is a virtual function that by 
	// default only calls the standard value function. The vector version can 
	// be overloaded in case a better definition possible.
	
	inline VariableType 
	InEqConstraintValue( Dimension i, const Variables & VariableValues )
	{ return Value( i, VariableValues ); }
	
	virtual ConstraintValues 
	InEqConstraintValue( const Variables & VariableValues )
	{ return Value( VariableValues ); }
			
public:
	
	// The constructor simply initializes the base class constraints
	
	InEqConstraints( void ) : Constraints()
	{}
	
	// It has a virtual destructor to ensure that derived classes will be 
	// properly destroyed when this base class is deleted.
	
	virtual ~InEqConstraints( void )
	{}
};

// The gradient vector constraint class simply extends the generic gradient 
// constraint class in the same way.

class GradientInEqConstraints : private GradientConstraints
{	
protected:

  // The declaration of the reference to the object holding the references 
  // to the constraint function and the gradient function can be reused. 
  
  using GradientConstraints::ConstraintGradientReferences;
  	
	// Adding constraints can again be done by the standard functions
	
	using GradientConstraints::Add;
	
	// The dimension function is again given as for the non-gradient 
	// constraints, and it can be overridden.
		
	virtual Dimension NumberOfInEqConstraints( void )
	{ return NumberOfConstraints(); }
	
	// The value functions are again defined similar to the non-gradient variant 
	
	inline VariableType 
	InEqConstraintValue( Dimension i, const Variables & VariableValues )
	{ return Value( i, VariableValues ); }
	
	virtual 
	ConstraintValues InEqConstraintValue( const Variables & VariableValues )
	{ return Value( VariableValues ); }

	// For the gradient, there are two options. One for the individual gradients
	// and this can be well handled by the generic function but it is aliased 
	// to make it explicit that it is an inequality constraint being computed,
	// but the matrix version can be re-defined if needed.
	
	inline GradientVector 
	InEqConstraintGradient( Dimension i, const Variables & VariableValues )
	{ return GradientConstraints::Gradient( i, VariableValues ); }
	
	virtual GradientMatrix 
	InEqConstraintGradient( const Variables & VariableValues )
	{ return GradientConstraints::Gradient( VariableValues ); }
	
public:
	
	// The constructor simply initialises the base class
	
	GradientInEqConstraints( void ) : GradientConstraints()
	{}
	
	// The destructor is virtual to ensure that base classes are properly 
	// destroyed.
	
	virtual ~GradientInEqConstraints( void )
	{}
};

/*==============================================================================

 Equality Constraints

==============================================================================*/
//
// Similar classes and functions are defined for the equality constraints. 
// They will reduce the dimensionality of the problem, and not many algorithms 
// support such constraints as they could be applied à priori and save some 
// variables.

class EqConstraints : private Constraints
{
protected:
	
  // The declaration of the reference to a constraint function is reused
  
  using Constraints::Reference;
  
	// Adding constraint functions implies calling a 
	
	using Constraints::Add;
	
	// The number of equality constraints must also be explicitly defined so that 
	// it is possible to refer to the number of constraints by class, and 
	// re-define it if necessary.
	
	virtual Dimension NumberOfEqConstraints( void )
	{ return Constraints::NumberOfConstraints(); }
	
	// The value functions are also in this case defined specific to the equality 
	// constraints, and with the possibility to overload the vector version.
	
	inline VariableType 
	EqConstraintValue( Dimension i, const Variables & VariableValues )
	{ return Value( i, VariableValues ); }
	
	virtual ConstraintValues 
	EqConstraintValue( const Variables & VariableValues )
	{ return Value( VariableValues ); }
	
public:
	
	// The constructor simply initializes the base class constraints
	
	EqConstraints( void ) : Constraints()
	{}
	
	// It has a virtual destructor to ensure that derived classes will be 
	// properly destroyed when this base class is deleted.
	
	virtual ~EqConstraints( void )
	{}
};

// The gradient version is similar

class GradientEqConstraints : public GradientConstraints
{	
protected:
	
  // The declaration of the reference to the object holding the references 
  // to the constraint function and the gradient function can be reused. 
  
  using GradientConstraints::ConstraintGradientReferences;
  
  // The functions to add the constraints are standard
	
	using GradientConstraints::Add;
	
  // The number of constraints are again specially defined for the equality 
  // constraints as above.

	virtual Dimension NumberOfEqConstraints( void )
	{ return GradientConstraints::NumberOfConstraints(); }  	
	
	// The value functions are similar to the non-gradient versions, with the 
	// possibility to re-define the vector version.
	
	inline VariableType 
	EqConstraintValue( Dimension i, const Variables & VariableValues )
	{ return Value( i, VariableValues ); }
	
	virtual ConstraintValues 
	EqConstraintValue( const Variables & VariableValues )
	{ return Value( VariableValues ); }
	
	// The gradient version are again similar to the inequality functions 
	// where the gradient matrix function can be re-defined.
	
	inline GradientVector 
	EqConstraintGradient( Dimension i, const Variables & VariableValues )
	{ return Gradient( i, VariableValues ); }
	
	virtual GradientMatrix 
	EqConstraintGradient( const Variables & VariableValues )
	{ return Gradient( VariableValues ); }
		
public:
	
	// The constructor simply initialises the base class
	
	GradientEqConstraints( void ) : GradientConstraints()
	{}
	
	// The destructor is virtual to ensure that base classes are properly 
	// destroyed.
	
	virtual ~GradientEqConstraints( void )
	{}
};

}      // End name space Optimization
#endif // OPTIMIZATION_CONSTRAINTS
