/*==============================================================================
Constraints

Constraints are more complicated since some algorithms accept all constraint 
values to be evaluated in one go as a vector, whereas others requires them to 
be evaluated individually. The C-style interface function request both the 
constraint values and the gradient of the constraints. The gradient is a null 
pointer if the algorithm does not need the constraint gradient. 

Given that the problem has n variables and m constraints, the gradient is a 
matrix with n * m elements. If C[j] is constraint j the column j consists of 
the partial derivatives of this with respect to each variable value x[i], in 
other words dC[j]/dx[i]. 

Author and Copyright: Geir Horn, 2018
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_NON_LINEAR_CONSTRAINTS
#define OPTIMIZATION_NON_LINEAR_CONSTRAINTS

#include <vector>											        // For variables and values
#include <sstream>                            // For error reporting
#include <stdexcept>                          // For standard exceptions
#include <memory>                             // Smart pointers
#include <armadillo>                          // Matrix library

#include <nlopt.h>                            // The C-style interface

#include "../Variables.hpp"                   // The dimension and value type
#include "../Constraints.hpp"                 // The generic constraint classes
#include "Algorithms.hpp"                     // Fundamental definitions

namespace Optimization::NonLinear
{
/*==============================================================================

 Vector valued constraints

==============================================================================*/
//
// It is necessary to distinguish between the cases where all the constraints 
// are computed in function call by a vector valued function and where they are 
// computed individually. The vector version is conceptually the simplest and 
// will be defined first.
//
// -----------------------------------------------------------------------------
// Vector constraint indirection mapper
// -----------------------------------------------------------------------------
//
// The constraint value computation function in NLOpt has a C-style signature 
// and takes a void data pointer assumed to be set to the 'this' pointer of the 
// relevant constraint class. It is strictly an internal class for the following 
// constraint computing classes as it has two virtual functions that must be 
// implemented by the constraint types to compute the required values.
//
// An interesting thing is that both the inequality constraints and the 
// equality constraints will derive from this class. If the class has a virtual
// method called 'ConstraintValues' it will create a problem when a problem 
// class has both inequality constraints and equality constraints because it is
// not possible to qualify a function name when overriding. In other words one 
// could not say 
// Inequality::ConstraintValues override
// Equality::ConstraintValues override
// because qualification is not allowed when overriding. Overriding just 
// ConstraintValues is ambiguous because it is not clear whether it is the 
// equality constraint function or the inequality constraint function that 
// will be overridden.
// 
// The solution is to define a generic "compute constraint" function and then 
// let the separate inequality and equality classes define their own named 
// virtual functions for the user classes to override. The "compute 
// constraint" function will then just call the separately named virtual 
// functions for each case. This is an additional indirection and hopefully 
// it will be optimised out - and memory wise and additional function pointer 
// is sacrificed.

class VectorConstraintsIndirectionMapper
{
protected:

	// --------------------------------------------------------------------------
	// Value producing functions
	// --------------------------------------------------------------------------
	//	
	// The simplest function is the one that returns the number of constraints 
	// and it must be defined by any constraint function providing class.
	
	virtual Dimension NumberOfConstraints( void ) = 0;
	
	// All constraint classes must be able to compute the constraints based on 
	// the variable values.
	
	virtual ConstraintValues
	ComputeConstraints( const Variables & VariableValues ) = 0;
	
	// However, not all classes must be able to compute the gradients of the 
	// constraints, and this is therefore defined as a second function that 
	// may call the gradient value function if that is supposed to exist, or 
	// throw an exception if the class is not supposed to have a constraint 
	// gradient but it is still invoked with a gradient pointer.
	// 
	// Given that the problem has n variables and m constraints, the gradient 
	// is a matrix with n * m elements. If C[j] is constraint j the column j 
	// consists of the partial derivatives of this with respect to each 
	// variable value x[i], in other words dC[j]/dx[i]. The reason for 
	// having the columns representing the constraints is because Armadillo 
	// stores the matrix in column order, and NLopt expects the value 
	// as one long vector with m blocks of n values. In other words, when 
	// reading out the matrix for NLopt column by column will be the most 
	// efficient for both packages.
	
	virtual GradientMatrix 
	ComputeGradients( const Variables & VariableValues ) = 0;

	// --------------------------------------------------------------------------
	// Mapper function
	// --------------------------------------------------------------------------
	//	
	// The actual indirection mapper is declared as a static function with the 
	// expected NLopt signature as static functions do not have the 'this' 
	// pointer and they can be used instead of C-functions. However, declaring 
	// them static class members allows the C++ access control to apply.
	
private:
	
	static void IndirectionMapper( unsigned int ConstraintSize,  // Dimensions 
																 double * ConstraintValues,    // Values
	 						                   unsigned int Size,            // problem size
																 const double *AssignedValues, // Variables
										             double *Gradient,             // Gradient
                                 void *Parameters )            // 'this' pointer
	{
		// The data pointer should in this case be the pointer to this mapper  
		// class, but since it is a void pointer a reinterpret cast must be used and 
		// it is not possible to check that this invariant holds.
		
		VectorConstraintsIndirectionMapper * This = 
				 reinterpret_cast< VectorConstraintsIndirectionMapper * >( Parameters );
		
		// Compute the values of the constraints
		
		std::vector< double > VariableValues( Size );
		
		for ( Dimension i = 0; i < Size; i++ )
			VariableValues[i] = AssignedValues[i];
		
		std::vector< double > Values = This->ComputeConstraints ( VariableValues	);
		
		if ( Values.size() == ConstraintSize )
			for ( Dimension i = 0; i < ConstraintSize; i++ )
				ConstraintValues[i] = Values[i];
		else
		{
			std::ostringstream ErrorMessage;
			
			ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
			             << ConstraintSize << " constraint values were expected, "
									 << "but only " << Values.size() << " values were computed.";
									 
		  throw std::logic_error( ErrorMessage.str() );
		}
		
		// The constraint gradient values are then computed if the gradient pointer
		// is given. For the derived classes supporting constraint gradients this 
		// will work as expected, otherwise a standard logical error will be thrown.
		
		if ( Gradient != nullptr )
		{
			GradientMatrix GradientValues = This->ComputeGradients( VariableValues );

			if ( ( GradientValues.n_cols == ConstraintSize ) && 
				   ( GradientValues.n_rows == Size ) )
		    for ( Dimension ConstraintIndex = 0; 
						  ConstraintIndex < ConstraintSize; ConstraintIndex++ )
			  {
					Dimension ConstraintOffset = ConstraintIndex * Size;
					
					for ( Dimension VariableIndex = 0; VariableIndex < Size; 
							  VariableIndex++ )
						Gradient[ ConstraintOffset + VariableIndex ] 
							= GradientValues( VariableIndex, ConstraintIndex );
				}
			else
			{
				std::ostringstream ErrorMessage;
				
				ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
				             << "A constraint gradient matrix of size "
										 << Size << " times " << ConstraintSize 
										 << "was expected, but a matrix of " 
										 << GradientValues.n_rows << " time " 
										 << GradientValues.n_cols << " values was computed";
										 
			  throw std::logic_error( ErrorMessage.str() );
			}
		}
	}
	
	// The indirection mapper function was defined as private to ensure that it
	// can only be called by the optimiser classes, and therefore these have to
	// be friends.
	
	template< Algorithm::ID >
	friend class Optimizer;

	// --------------------------------------------------------------------------
	// Setting the constraint function for solvers
	// --------------------------------------------------------------------------
	//	
	// There are support functions for setting the inequality constraints and 
	// the equality constraints for a given solver. These are defined here as 
	// they are generic for both non-gradient and gradient constraints.
	//
	// The function to set the inequality constraint function for a given solver
	// takes a vector of tolerance values and register the indirection mapper 
	// to handle the vector valued constraint evaluation. 
	// 
	// It is important that the size of the tolerance vector equals the number of
	// constraint values, otherwise it will produce an invalid argument exception.
  // If the function is called when no constraints are defined, it will throw 
  // a logical error exception. This means that algorithms for which the 
  // constraints are optional must test for this situation prior to calling 
  // this function. 

protected:
  
	inline
	nlopt_result SetInEqConstraints( SolverPointer Solver, 
													         const std::vector< double > & Tolerances )
	{
		Dimension ConstraintCount = NumberOfConstraints();
		
		if ( ConstraintCount > 0 )
    {
      if ( Tolerances.size() == ConstraintCount )
 		    return nlopt_add_inequality_mconstraint( Solver, ConstraintCount, 
               &IndirectionMapper, this, Tolerances.data() );
      else
      {
        std::ostringstream ErrorMessage;
				
				ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
				             << "The vector of tolerances given for the inequality "
                     << "constraints has " << Tolerances.size() 
                     << " elements and it should have had one for each of the " 
                     << ConstraintCount << " defined constraints";
										 
			  throw std::invalid_argument( ErrorMessage.str() );
      }  
    }
		else
		{
			std::ostringstream ErrorMessage;
			
			ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
							     << "Setting the inequality constraints for a solver is "
									 << "not possible when no constraints are defined.";
									 
		  throw std::logic_error( ErrorMessage.str() );
		}
	}
  
  inline
	nlopt_result SetInEqConstraints( SolverPointer Solver, double Tolerance )
  {
    return SetInEqConstraints( Solver, 
           std::vector< double >( NumberOfConstraints(), Tolerance ) );
  }
	
	// The function to set the equality constraints are similar to the function 
	// for the inequality constraints by taking a set of tolerances that must 
	// equal the number of equality constraints to be provided.
	
	inline
	nlopt_result SetEqConstraints( SolverPointer Solver, 
												         const std::vector< double > & Tolerances )
	{
		Dimension ConstraintCount = NumberOfConstraints();
		
		if ( ConstraintCount > 0 )
    {
      if ( Tolerances.size() == ConstraintCount )
        return nlopt_add_equality_mconstraint( Solver, ConstraintCount, 
						   &IndirectionMapper, this, Tolerances.data() );
      else
      {
        std::ostringstream ErrorMessage;
				
				ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
				             << "The vector of tolerances given for the equality "
                     << "constraints has " << Tolerances.size() 
                     << " elements and it should have had one for each of the " 
                     << ConstraintCount << " defined constraints";
										 
			  throw std::invalid_argument( ErrorMessage.str() );
      }  
    }
		else
		{
			std::ostringstream ErrorMessage;
			
			ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
							     << "Setting the equality constraints for a solver is "
									 << "not possible when no constraints are defined.";
									 
		  throw std::logic_error( ErrorMessage.str() );
		}
	}
  
  inline
	nlopt_result SetEqConstraints( SolverPointer Solver, double Tolerance )
  {
    return SetEqConstraints( Solver, 
           std::vector< double >( NumberOfConstraints(), Tolerance ) );
  }
	
	// --------------------------------------------------------------------------
	// Destructor
	// --------------------------------------------------------------------------
	//	
	// The constructor does nothing and the class has a virtual constructor to 
	// ensure that all derived classes are properly destructed.
	
public:
	
	virtual ~VectorConstraintsIndirectionMapper( void )
	{}
};

// -----------------------------------------------------------------------------
// Inequality constraints
// -----------------------------------------------------------------------------
//
// A standard optimization problem has constraints of the form g(x) <= 0 
// and the constraint class basically is the interface to define the 
// left hand side function. Note that this is a vector function with as many 
// elements as there are constraints, and that there can be many more 
// constraints than there are variables. The function must therefore return a 
// vector of the constraint values evaluated at the given variable values.

class InEqConstraints 
: public Optimization::InEqConstraints,
  private VectorConstraintsIndirectionMapper
{
private:
	
	// The number of constraints are given by the virtual function of the 
	// equality constraint class, and the one expected by the mapper class 
	// could create confusion as the mapper is used both for inequality 
	// constraints and equality constraints
	
	virtual Dimension NumberOfConstraints( void ) final
	{ return NumberOfInEqConstraints(); }

	// The function computing the constraints inherited from the indirection 
	// mapper simply calls this function, and it should not be further 
	// overloaded.
	
	virtual	ConstraintValues 
	ComputeConstraints( const Variables & VariableValues ) final
  {
		return InEqConstraintValue( VariableValues );
	}
	
	// If the compute gradient function is called, then it is a sign that the 
	// wrong constraint class is used and an error should be thrown indicating 
	// that there is no support for gradients for the basic Inequality class.
	
	virtual GradientMatrix 
	ComputeGradients( const Variables & VariableValues ) final
	{
		std::ostringstream ErrorMessage;
		
		ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
	               << "Constraint gradients were expected for the inequality "
								 << "constraints when the plain Inequality constraint "
								 << "class was used";
								 
	  throw std::logic_error( ErrorMessage.str() );		
	}
	
	// The functions to set the constraint mappers for a given solver is reused,
	// but only the version for the inequality constraints, and only for derived 
	// classes
	
protected:
	
	using VectorConstraintsIndirectionMapper::SetInEqConstraints;
				
public:
	
	virtual ~InEqConstraints( void )
	{}
};

// If the algorithm requires the gradient of the constraint values to be given
// the optimiser class should inherit the constraint class requiring 
// a function to compute the gradient values of the constraints evaluated for 
// the given variables.

class GradientInEqConstraints 
: public Optimization::GradientInEqConstraints,
  private VectorConstraintsIndirectionMapper
{
private:
	
	// The number of constraints and the value of the constraints are defined 
	// based on the base class definitions similar to the non-gradient version
	
	virtual Dimension NumberOfConstraints( void ) final
	{ return NumberOfInEqConstraints(); }

	virtual	ConstraintValues 
	ComputeConstraints( const Variables & VariableValues ) final
  {
		return InEqConstraintValue( VariableValues );
	}
	
  // The gradient computation function is in this case simply delegating 
	// the computation to the general function
	
	virtual GradientMatrix 
	ComputeGradients( const Variables & VariableValues ) final
	{
		return InEqConstraintGradient( VariableValues );
	}	
	
	// The function to set the constraints for the solver is again made accessible 
	// for derived classes
	
protected:
	
  using VectorConstraintsIndirectionMapper::SetInEqConstraints;
	
public:
	
	virtual ~GradientInEqConstraints( void )
	{}
};

// -----------------------------------------------------------------------------
// Equality constraints
// -----------------------------------------------------------------------------
//
// These are defined analogous to the inequality constraint classes

class EqConstraints 
: public Optimization::EqConstraints,
  private VectorConstraintsIndirectionMapper
{
private:
	
  // The number of constraints required by the mapper class is provided by 
  // the equality constraints, although this can be overridden.

  virtual Dimension NumberOfConstraints( void ) final
	{ return NumberOfEqConstraints(); }
		
	// The function computing the constraints inherited from the indirection 
	// mapper simply calls the function defined for the equality constraints
	
	virtual ConstraintValues 
	ComputeConstraints( const Variables & VariableValues ) final
  { return EqConstraintValue( VariableValues );	}
	
	// If the compute gradient function is called, then it is a sign that the 
	// wrong constraint class is used and an error should be thrown indicating 
	// that there is no support for gradients for the basic Inequality class.
	
	virtual GradientMatrix 
	ComputeGradients( const Variables & VariableValues ) final
	{
		std::ostringstream ErrorMessage;
		
		ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
	               << "Constraint gradients were expected for the equality "
								 << "constraints when the plain Equality constraint "
								 << "class was used";
								 
	  throw std::logic_error( ErrorMessage.str() );		
	}
	
	// The class also re-uses the functions to set the constraints, and it offers
	// only the equality version of the functions.
	
protected:

  using VectorConstraintsIndirectionMapper::SetEqConstraints;
				
public:
	
	virtual ~EqConstraints( void )
	{}
};

// If the algorithm requires the gradient of the constraint values to be given
// the optimiser class should inherit the constraint class requiring 
// a function to compute the gradient values of the constraints evaluated for 
// the given variables.

class GradientEqConstraints 
: public Optimization::GradientEqConstraints,
  private VectorConstraintsIndirectionMapper
{
private:
	
	// Similar to the previous classes, it overloads privately the value functions
	// required by the mapper class using the equality constraint class' functions
	
	virtual Dimension NumberOfConstraints( void ) final
	{ return NumberOfEqConstraints(); }
	
	// The values are readily computed by the value function
	
	virtual ConstraintValues 
	ComputeConstraints( const Variables & VariableValues ) final
  { return EqConstraintValue( VariableValues );	}
	
  // The gradient function should be defined by one of the earlier classes or 
  // the basic equality gradient function should be overloaded if direct 
  // definitions of the gradient matrix is possible.
  
  virtual GradientMatrix 
	ComputeGradients( const Variables & VariableValues ) final
	{ return EqConstraintGradient( VariableValues ); }
		
protected:
		
	// The functions to set the constraints for the particular solver is imported 
	// from the mapper class.
	
  using VectorConstraintsIndirectionMapper::SetEqConstraints;
	
public:
	
	virtual ~GradientEqConstraints( void )
	{}
};

/*==============================================================================

 Individually valued constraints

==============================================================================*/
//
// The same 5 classes are defined also for the individually valued constraints,
// with the main differences being in the indirection mapper function and in the 
// argument lists for the functions as they now need to take the index ID of the 
// constraint to evaluate.
//
// -----------------------------------------------------------------------------
// Individual constraint indirection mapper
// -----------------------------------------------------------------------------
//
// For a problem with vector constraints, the mapper was registered only once 
// and it was sufficient to provide the 'this' pointer in order to evaluate the 
// constraints. When evaluating individual constraints, it is not only necessary 
// to pass the 'this' pointer, but also to pass the index of the constraint to 
// be evaluated. Thus, the 'this' pointer and the index are combined into one 
// object whose pointer is passed to the mapper function.
//
// This poses a minor problem in that the pointer to the data object with the 
// 'this' pointer and the constraint index is stored in the solver object when 
// the indirection mapper is registered as the constraint function. The memory 
// must stay allocated for this object until the constraint is removed, and 
// it should also be observed that constraints can only be added - and they 
// must be deleted when a new solver object is initialised, or if they are 
// explicitly deleted.
//
// The solution is to use specialized 'functors' referring to a particular 
// constraint, and pass this pointer to the mapper function. Since the 
// user defined constraints do not change, each derived class can maintain 
// their own functor objects. However, both the constraint functions and 
// gradient functions are already stored in the vectors of the base classes,
// and it is only necessary to reference these. 

class IndividualConstraintIndirectionMapper
{
	// --------------------------------------------------------------------------
	// Constraint references
	// --------------------------------------------------------------------------
	//	
	// Since the mapper function does not distinguish between non-gradient 
	// algorithms and gradient algorithms, both functions must be available 
	// even if the latter is not used. A two-level reference is implemented 
	// to avoid always testing value of the gradient reference.

private:
		
	class ConstraintReference
	{
	private:
		
		Constraints::Reference TheConstraintFunction;
		
	public:
		
		inline VariableType Value( const Variables & VariableValues )
		{ return (*TheConstraintFunction)( VariableValues ); }
		
		virtual GradientVector Gradient( const Variables & VariableValues )
		{
			std::ostringstream ErrorMessage;
			
			ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
			             << "Constraint gradients are required when they "
									 << "are not defined";
									 
		  throw std::logic_error( ErrorMessage.str() );
		}
		
		ConstraintReference( const Constraints::Reference & TheFunction )
		: TheConstraintFunction( TheFunction )
		{}
		
		ConstraintReference( void ) = delete;
	};
	
	// Then this is extended with a reference to the gradient function 
	
	class GradientConstraintReference : public ConstraintReference
	{
	private:
		
		GradientConstraints::GradientReference TheConstraintGradient;
		
	public:
		
		virtual GradientVector Gradient( const Variables & VariableValues )
		{ return (*TheConstraintGradient)( VariableValues ); }
		
		// The constructor requires references both to the constraint function
		// and to the gradient function
		
		GradientConstraintReference(
			const Constraints::Reference                 & TheFunction,
			const GradientConstraints::GradientReference & TheGradient )
		: ConstraintReference  ( TheFunction ),
		  TheConstraintGradient( TheGradient )
		{}
		
		GradientConstraintReference( void ) = delete;
	};
	
	// These references are stored in a private vector, and adding the 
	// constraints to the solver will fail if this vector is empty.
	
	std::vector< std::unique_ptr< ConstraintReference > > References;
	
	// The references are stored by derived classes by store functions that 
	// creates the relevant class depending on which function used.
	
protected:
	
	inline void Store( const Constraints::Reference & TheConstraintFunction )
	{
		References.emplace_back( 
			std::make_unique< ConstraintReference >( TheConstraintFunction ) );
	}
	
	inline void Store( 
				 const Constraints::Reference & TheConstraintFunction,
				 const GradientConstraints::GradientReference & TheGradientFunction )
	{
		References.emplace_back( 
			std::make_unique< GradientConstraintReference >( TheConstraintFunction, 
																											 TheGradientFunction ) );
	}

	// --------------------------------------------------------------------------
	// Mapper function
	// --------------------------------------------------------------------------
	//
	// The mapper function takes a pointer to a constraint reference as data. 
	// it takes the number of variables, a vector of variable values, a vector 
	// of gradient values, and the parameter record pointer. The gradient value
	// can be NULL if the algorithm does not require a gradient, and the 
	// parameter pointer is the constraint reference pointer.

	static double IndirectionMapper( unsigned int   NumberOfVariables, 
																	 const double * VariableValues, 
																	 double       * GradientValues, 
																	 void         * Parameters)
	{
		// Getting the constraint function
		
		ConstraintReference * 
		TheConstraint = reinterpret_cast< ConstraintReference * >( Parameters );
		
		// Converting the given C-style variable value vector to a proper STL
		// vector
		
		Variables GivenVariables( NumberOfVariables );
		
		for ( Dimension i = 0; i < NumberOfVariables; i++ )
			GivenVariables[ i ] = VariableValues[ i ];
		
		// If the gradient values are requested, they are computed and set into 
		// the gradient vector.
		
		if ( GradientValues != nullptr )
		{
			GradientVector TheGradient = TheConstraint->Gradient( GivenVariables );
			
			for ( Dimension i = 0; i < NumberOfVariables; i++ )
				GradientValues[ i ] = TheGradient[ i ];
		}
		
		// The constraint value can then be returned
		
		return TheConstraint->Value( GivenVariables );
	}

	// --------------------------------------------------------------------------
	// Setting the constraint function for solvers
	// --------------------------------------------------------------------------
	//	
  // The interface functions to set the individual gradient functions for a 
  // solver is identical to the vector version, however they will register the
  // mapper function for each of the constraints.

protected:
	
 	inline
	nlopt_result SetInEqConstraints( SolverPointer Solver, 
													         const std::vector< double > & Tolerances )
	{
		if ( References.size() > 0 )
		{
			if ( References.size() == Tolerances.size() )
				for ( Dimension i = 0; i < References.size(); i++ )
				{
					nlopt_result 
					Result = nlopt_add_inequality_constraint( Solver, &IndirectionMapper,
																									  References[ i ].get(), 
																									  Tolerances[ i ] );
					if ( Result < 0 )
						return Result;
				}
			else
		  {
				std::ostringstream ErrorMessage;
				
				ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
				             << "The vector of tolerances given when setting the "
										 << "inequality constraints has " 
										 << Tolerances.size() << " elements and it should have "
										 << "one for each of the " << References.size()
										 << " defined constraints";
										 
			  throw std::invalid_argument( ErrorMessage.str() );
			}
		}
		else
		{
			std::ostringstream ErrorMessage;
			
			ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
							     << "Setting the inequality constraints for a solver is "
									 << "not possible when no constraints are defined.";
									 
		  throw std::logic_error( ErrorMessage.str() );
		}
		
		return NLOPT_SUCCESS;
	}

  // If all the tolerances should be set to the same value, an overloaded 
	// version of this function can be used which delegates the setting of the 
	// indirection mapper to the previous function.
	
	inline nlopt_result SetInEqConstraints( SolverPointer Solver, 
																				  double ToleranceValue )
	{
		return SetInEqConstraints( Solver, 
			     std::vector< double >( References.size(), ToleranceValue ) );
	}

		// The function to set the equality constraints are similar to the function 
	// for the inequality constraints by taking a set of tolerances that must 
	// equal the number of equality constraints to be provided.
	
	inline
	nlopt_result SetEqConstraints( SolverPointer Solver, 
												         const std::vector< double > & Tolerances )
	{
		
		if ( References.size() > 0 )
		{
			if ( References.size() == Tolerances.size() )
				for ( Dimension i = 0; i < References.size(); i++ )
				{
					nlopt_result 
					Result = nlopt_add_equality_constraint( Solver, &IndirectionMapper,
																								  References[ i ].get(), 
																								  Tolerances[ i ] );
					if ( Result < 0 )
						return Result;
				}
			else
		  {
				std::ostringstream ErrorMessage;
				
				ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
				             << "The vector of tolerances given when setting the "
										 << "equality constraints has " 
										 << Tolerances.size() << " elements and it should have "
										 << "one for each of the " << References.size()
										 << " defined constraints";
										 
			  throw std::invalid_argument( ErrorMessage.str() );
			}
		}
		else
		{
			std::ostringstream ErrorMessage;
			
			ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
							     << "Setting the equality constraints for a solver is "
									 << "not possible when no constraints are defined.";
									 
		  throw std::logic_error( ErrorMessage.str() );
		}
	}
	
	// The alternative function takes the number of constraints and a common 
	// tolerance value and sets this for all constraints to be evaluated.
	
	inline nlopt_result SetEqConstraints( SolverPointer Solver, 
																			  double ToleranceValue )
	{
		return SetEqConstraints( Solver, 
           std::vector< double >( References.size(), ToleranceValue ) );
	}
};

// -----------------------------------------------------------------------------
// Inequality constraints
// -----------------------------------------------------------------------------
//
// The user level classes binds the general constraint class with the indirection
// mapper, and make sure that only the inequality constraints can be set, and 
// that every constraint added will be stored in the indirection mapper. A further 
// point is that the classes working with individual constraints will only allow 
// individual constraints to be used. 

class IndividualInEqConstraints 
: public Optimization::InEqConstraints,
  private IndividualConstraintIndirectionMapper
{
private:
	
	// The vector version of the inequality constraints are defined to be an error 
	// message only.
	
	virtual ConstraintValues 
	InEqConstraintValue( const Variables & VariableValues ) final
	{
		std::ostringstream ErrorMessage;
		
		ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
		             << "The vector constraint function should not be called when "
								 << " the individual inequality constraints are required";
								 
	  throw std::logic_error( ErrorMessage.str() );
	}
	
protected:

	// The number of constraints can no longer be defined independent of the 
	// individual constraints added.
	
	virtual Dimension NumberOfInEqConstraints( void ) final
	{ return Optimization::InEqConstraints::NumberOfInEqConstraints();  }
		
	// The Add function is overloaded to pass the constraint function also to the 
	// indirection mapper.
	
	inline Reference Add( const Constraint & TheConstraintFunction )
	{
		Reference NewConstraint = 
								 Optimization::InEqConstraints::Add( TheConstraintFunction );
								 
	  Store( NewConstraint );
		
		return NewConstraint;
	}
	
	// Then the function to store the inequality constraints are exported so that 
	// derived classes can use it to define the constraints with the solver for 
	// a problem.
	
	using IndividualConstraintIndirectionMapper::SetInEqConstraints;
};

// The individual gradient constraints are similar except that it requires two 
// functions to be given to the function to add constraints, and stores both 
// functions.

class IndividualGradientInEqConstraints 
: public Optimization::GradientInEqConstraints,
  private IndividualConstraintIndirectionMapper
{
private:

	// The vector functions for obtaining the gradient values and the gradient 
	// matrix are disabled.
	
	virtual ConstraintValues 
	InEqConstraintValue( const Variables & VariableValues ) final
	{
		std::ostringstream ErrorMessage;
		
		ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
		             << "The vector constraint function should not be called when "
								 << " the individual inequality constraints are required";
								 
	  throw std::logic_error( ErrorMessage.str() );		
	}	
	
	virtual GradientMatrix 
	InEqConstraintGradient( const Variables & VariableValues ) final
	{
		std::ostringstream ErrorMessage;
		
		ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
		             << "The gradient matrix function should not be called when "
								 << " the individual inequality constraints are required";
								 
	  throw std::logic_error( ErrorMessage.str() );				
	}
	
protected:
	
	// The constraint count must equal the defined constraints.
	
	virtual Dimension NumberOfInEqConstraints( void ) final
	{ return Optimization::GradientInEqConstraints::NumberOfInEqConstraints(); }
	
	// The signature of the function adding constraint functions requires both 
	// functions to be given
	
	inline ConstraintGradientReferences
	Add( const Constraint & TheConstraintFunction, 
		   const ConstraintGradient & TheGradientFunction )
	{
		ConstraintGradientReferences NewFunctions = 
		Optimization::GradientInEqConstraints::Add( TheConstraintFunction, 
																								TheGradientFunction );
		
		Store( NewFunctions.TheConstraintReference, 
           NewFunctions.TheGradientReference );
		
		return NewFunctions;
	}
	
	// Finally the functions for setting the constraints for a particular solver 
	// can be imported in the inequality version.
	
	using IndividualConstraintIndirectionMapper::SetInEqConstraints;
};

// -----------------------------------------------------------------------------
// Equality constraints
// -----------------------------------------------------------------------------
//
// The equality constraints follow the same pattern ensuring to erase the 
// vector valued functions and overriding the functions to add the constraint 
// functions. 

class IndividualEqConstraints 
: public Optimization::EqConstraints,
  private IndividualConstraintIndirectionMapper
{
private:

  virtual ConstraintValues 
	EqConstraintValue( const Variables & VariableValues ) final
	{ 
		std::ostringstream ErrorMessage;
		
		ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
		             << "The vector constraint function should not be called when "
								 << " the individual equality constraints are required";
								 
	  throw std::logic_error( ErrorMessage.str() );		
	}	
	
protected:
	
	virtual Dimension NumberOfEqConstraints( void ) final
	{ return Optimization::EqConstraints::NumberOfEqConstraints(); }
	
	// Adding a constraint is basically using the already defined functionality 
	// and storing the result 
		
	inline Reference Add( const Constraint & TheConstraintFunction )
  {
		Reference NewConstraint = EqConstraints::Add( TheConstraintFunction );
													 
	  Store( NewConstraint );
		
		return NewConstraint;
	}
	
	// The function to set the constraint for the solver to use is provided by 
	// the mapper class. 
	
	using IndividualConstraintIndirectionMapper::SetEqConstraints;
};

// The gradient version is similar except that it also has to disable the 
// gradient matrix computation, and store both the constraint function and the 
// constraint gradient function

class IndividualGradientEqConstraints 
: public Optimization::GradientEqConstraints,
  private IndividualConstraintIndirectionMapper
{
private:
	
	virtual ConstraintValues 
	EqConstraintValue( const Variables & VariableValues ) final
	{ 
		std::ostringstream ErrorMessage;
		
		ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
		             << "The vector constraint function should not be called when "
								 << " the individual equality constraints are required";
								 
	  throw std::logic_error( ErrorMessage.str() );				
	}
	
	virtual GradientMatrix 
	EqConstraintGradient( const Variables & VariableValues ) final
	{
		std::ostringstream ErrorMessage;
		
		ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
		             << "The gradient matrix function should not be called when "
								 << " the individual equality constraints are required";
								 
	  throw std::logic_error( ErrorMessage.str() );						
	}
	
protected:
	
	virtual Dimension NumberOfEqConstraints( void ) final
	{ return Optimization::GradientEqConstraints::NumberOfEqConstraints(); }
	
	// Adding individual constraints will also store their references for 
	// registration with the solver
	
	inline ConstraintGradientReferences 
  Add( const Constraint & TheConstraintFunction, 
		   const ConstraintGradient & TheGradientFunction )
	{
		ConstraintGradientReferences NewFunctions = 
		Optimization::GradientEqConstraints::Add( TheConstraintFunction, 
																							TheGradientFunction );
		
		Store( NewFunctions.TheConstraintReference, 
           NewFunctions.TheGradientReference );
		
		return NewFunctions;
	}
	
	// The registered constraints can be bound to the solver by the function to
	// set the equality constraints.
	
	using IndividualConstraintIndirectionMapper::SetEqConstraints;
};

}      // Name space Optimization non-linear
#endif // OPTIMIZATION_NON_LINEAR_CONSTRAINTS
