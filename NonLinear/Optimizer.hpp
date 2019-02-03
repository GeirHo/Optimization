/*==============================================================================
Optimizer

The Optimizer is the object specialised for a given algorithm. It is derived
from the optimizer interface defined here offering error checking of the results
and defining the virtual functions of the optimizer interface.

Author and Copyright: Geir Horn, 2018-2019
License: LGPL 3.0
==============================================================================*/

#ifndef OPTIMIZATION_NON_LINEAR_OPTIMIZER
#define OPTIMIZATION_NON_LINEAR_OPTIMIZER

#include <cerrno>                            // System error codes
#include <chrono>                            // Search time limit in seconds
#include <map>                               // To ignore some errors
#include <optional>                          // For values that may not be set
#include <string>                            // Strings
#include <system_error>                      // Error categories
#include <vector>                            // For variables and values

#include <boost/numeric/conversion/cast.hpp> // Casting numeric types
#include <nlopt.h>                           // The C-style interface

#include "../Variables.hpp"                  // Basic definitions
#include "NonLinear/Algorithms.hpp"          // Definition of the algorithms
#include "NonLinear/Objective.hpp"           // Objective function

namespace Optimization::NonLinear
{

// The optimizer is a class specialised on the various algorithm types,
// and its generic signature is stated here.

template< Algorithm::ID > class Optimizer;

// -----------------------------------------------------------------------------
// Optimizer interface
// -----------------------------------------------------------------------------
//
// The optimiser interface defines methods that must be supported by all the
// various algorithmic variants specialised at the bottom of this file.
// The basic functionality is to create the solver, clean the solver, and
// find an optimal solution. The interface is an abstract class that should
// only be used as a base class for the derived methods

class OptimizerInterface : virtual public Objective
{
private:
  // The various NLOpt functions return a status code that by default will
  // result in an error being thrown. However, the action is controlled by
  // a map where some of the error codes may be mapped to success and thereby
  // ignored. By default all errors will result in an exception.

  std::map< nlopt_result, nlopt_result > StatusAction;

  // ---------------------------------------------------------------------------
  // Managing the solver
  // ---------------------------------------------------------------------------
  //
  // Each optimizer class has a solver.

  SolverPointer Solver;

protected:

  // Derived classes should define the algorithms they support in case it is
  // necessary to check this later.

  virtual Algorithm::ID GetAlgorithm( void ) = 0;

  // There is a small helper function that returns the algorithms as a string.

  inline std::string GetAlgorithmName( void )
  {
    return std::string( 
    nlopt_algorithm_name( static_cast< nlopt_algorithm >( GetAlgorithm() )));
  }

  // There is another helper to get the dimension of an allocated solver or
  // zero if no solver has been allocated.

  inline Dimension GetDimension( void )
  {
    if( Solver != nullptr )
      return nlopt_get_dimension( Solver );
    else
      return 0;
  }

  // The solver can be cleared by another utility. It is not necessary to call
  // this as setting a new solver will be sufficient. However, if one wants to
  // use a solver only for the period the problem is solved, it is possible to
  // explicitly clear the solver.

  inline void DeleteSolver( void )
  {
    if ( Solver != nullptr ) 
    {
      nlopt_destroy( Solver );
      Solver = nullptr;
    }
  }

  // The function to set up the solver must be overloaded by the algorithm
  // specific specialisations of the optimizer because the solver initialisation
  // depends on the type of algorithm, in particular the type of constraints
  // supported by the algorithm. This version is just responsible for creating
  // the solver, and should be called by the overloaded algorithmic specific
  // functions.

  virtual SolverPointer CreateSolver( Dimension NumberOfVariables, 
                        Objective::Goal Direction = Objective::Goal::Minimize )
  {
    if( Solver != nullptr )
      DeleteSolver();

    if ( GetAlgorithm() < Algorithm::ID::MaxNumber )
      Solver = nlopt_create( static_cast< nlopt_algorithm >( GetAlgorithm() ), 
                             NumberOfVariables );
    else
    {
      std::ostringstream ErrorMessage;
      
      ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
                   << "The algorithm ID of this class does not correspond "
                   << "to a legal algorithm";
                   
      throw std::invalid_argument( ErrorMessage.str() );
    }

    if ( Solver != nullptr )
      return Solver;
    else 
    {
      std::ostringstream ErrorMessage;

      ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
                   << "Failed to allocate NLopt solver";

      throw std::runtime_error(ErrorMessage.str());
    }
  }

  // ---------------------------------------------------------------------------
  // Error handling
  // ---------------------------------------------------------------------------
  //
  // The various NLOpt functions return a status code indicating the outcome of
  // an operation and the error codes are converted to an exception by the
  // status checking function.
  //
  // Some of these errors may be critical while some of these are more like
  // recoverable issues. It may therefore be desired to catch and ignore the
  // less critical ones Thus there are functions to ignore and react on a
  // given status. Note that the 'at' function is used to retrieve the
  // status record as this will throw if an illegal status is provided.

  inline void IgnoreStatus( nlopt_result Status )
  {
    StatusAction.at( Status ) = NLOPT_SUCCESS;
  }

  inline void ThrowStatus( nlopt_result Status )
  {
    StatusAction.at( Status ) = Status;
  }

  // If a status throws, then it should be possible to catch selectively
  // the exception to make the right action. There are several runtime
  // errors, and it is therefore necessary to define derived classes to
  // indicate which error that raised the exception. They are all derived
  // from the standard runtime error, and one should be able to catch
  // the as a group if needed.
  //
  // The first of these is the exception thrown if there is a problem
  // with the solution accuracy by the solver being constrained by
  // a round-off error.

  class RoundoffLimited : public std::runtime_error
  {
  public:
  
    RoundoffLimited( const std::string & ErrorMessage )
    : std::runtime_error( ErrorMessage )
    {}

    RoundoffLimited(void) = delete;
  };

  // The second class is thrown if the user aborted the algorithm
  // by a forced stop. In this case the user should know the reason
  // for the exception.

  class ForcedStop : public std::runtime_error
  {
  public:
  
    ForcedStop( const std::string & ErrorMessage )
    : std::runtime_error( ErrorMessage )
    {}

    ForcedStop( void ) = delete;
  };

  // it is not expected that the application will throw on a
  // successful event. If this is not wanted, the corresponding
  // events should be muted. However, if one needs to do special
  // handling in case of a conditional success, there is a special
  // base class for these exceptions.

  class ConditionalSuccess
  {
  private:
  
    const std::string Explaination;

  public:
  
    inline std::string what( void )
    { return Explaination; }

    ConditionalSuccess( const std::string & Description )
    : Explaination( Description )
    {}

    ConditionalSuccess( void ) = delete;
  };

  // It is possible to set limit value for terminating the search.
  // For a minimization this is the maximum vale allowed for the
  // objective function, while for the maximization it is the least
  // value required for the objective function. A stop value is
  // useful if one iterative seek a solution by deploying the
  // optimiser on different sub region in a branch-and-bound style,
  // which may be a reason to want this

  class StopValueReached : public ConditionalSuccess
  {
  public:
  
    StopValueReached( const std::string & Description )
    : ConditionalSuccess( Description )
    {}

    StopValueReached( void ) = delete;
  };

  // If one of the tolerances stopped the evaluation, then tolerance
  // reached classed will be thrown,

  class ToleranceReached : public ConditionalSuccess
  {
  public:
  
    ToleranceReached( const std::string & Description) 
    : ConditionalSuccess( Description )
    {}

    ToleranceReached( void ) = delete;
  };

  // In the same way there is a class to indicate that a limit on the
  // number of evaluations or the time out was reached.

  class LimitReached : public ConditionalSuccess
  {
  public:
  
    LimitReached( const std::string & Description )
    : ConditionalSuccess( Description )
    {}

    LimitReached( void ) = delete;
  };

  // The function to check and throw upon a status different from
  // success optionally takes a string describing the execution
  // context, i.e. in which situation resulted in the given status.
  // The aim os to give better error messages.

  void CheckStatus( const nlopt_result Status, 
                    const std::string& Context = std::string() )
  {
    switch( StatusAction.at( Status ) ) 
    {
    case NLOPT_FAILURE: 
      {
        std::ostringstream ErrorMessage;

        ErrorMessage << "General failure when performing operation " << Context;

        throw std::runtime_error( ErrorMessage.str() );
      } 
      break;
    case NLOPT_INVALID_ARGS: 
      {
        std::ostringstream ErrorMessage;

        ErrorMessage << "A function call " << Context 
                     << " was performed with invalid arguments for the "
                     << "algorithm " << GetAlgorithmName();

        throw std::invalid_argument( ErrorMessage.str() );
      } 
      break;
    case NLOPT_OUT_OF_MEMORY:
      // Ideally a bad_alloc exception should be thrown but it does
      // not allow for a descriptive message, and it is also not
      // sure that it really was bad allocation as it could be another
      // memory related issue. A standard system error is therefore
      // generated instead.

      throw std::system_error(ENOMEM, std::generic_category(), Context);
      break;
    case NLOPT_ROUNDOFF_LIMITED: 
      {
        std::ostringstream ErrorMessage;

        ErrorMessage << "The operation " << Context 
                     << " was round off limited";

        throw RoundoffLimited( ErrorMessage.str() );
      } 
      break;
    case NLOPT_FORCED_STOP: 
      {
        std::ostringstream ErrorMessage;

        ErrorMessage << Context << " resulted in a forced user generated"
                     << " stop request";

        throw ForcedStop( ErrorMessage.str() );
      } 
      break;
    case NLOPT_SUCCESS:
      break; // Easy, do nothing!
    case NLOPT_STOPVAL_REACHED:
      throw StopValueReached( Context );
      break;
    case NLOPT_FTOL_REACHED: 
      {
        std::ostringstream StopInformation;

        StopInformation << "The tolerance on the objective function was "
                        << "reached " << Context;

        throw ToleranceReached( StopInformation.str() );
      } 
      break;
    case NLOPT_XTOL_REACHED: 
      {
        std::ostringstream StopInformation;

        StopInformation << "The tolerance on the variable value changes "
                        << "was reached " << Context;

        throw ToleranceReached( StopInformation.str() );
      } 
      break;
    case NLOPT_MAXEVAL_REACHED: 
      {
        std::ostringstream StopInformation;

        StopInformation << "The limit on the number of evaluations was "
                        << "reached " << Context;

        throw LimitReached( StopInformation.str() );
      } 
      break;
    case NLOPT_MAXTIME_REACHED: 
      {
        std::ostringstream StopInformation;

        StopInformation << "The time out for the execution time was "
                        << "reached " << Context;

        throw LimitReached( StopInformation.str() );
      } 
      break;
    default: 
      {
        // This should never be evaluated because the "at" function
        // used to map the status to the status action should throw
        // if the status was unknown.

        std::ostringstream ErrorMessage;

        ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
                     << "An unknown NLopt return status" << Status 
                     << " was provided indicating a serious misuse";

        throw std::invalid_argument( ErrorMessage.str() );
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Stopping criteria
  // ---------------------------------------------------------------------------
  //
  // There are two functions that can be used to set or read the stop value,
  // and it should be noted that this will not return a value if the solver has
  // has not been allocated or if the value is not effective and set to a huge
  // maximum value.

  inline void StopValue( double TheValue )
  {
    if ( Solver != nullptr ) 
    {
      nlopt_result Result = nlopt_set_stopval( Solver, TheValue );
      CheckStatus( Result, "Setting a stop value for the solver" );
    } 
    else 
    {
      std::ostringstream ErrorMessage;

      ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
                   << "Trying to set the stop value for the solver "
                   << "before the solver has been created";

      throw std::logic_error(ErrorMessage.str());
    }
  }

  inline std::optional< double > StopValue( void )
  {
    std::optional< double > TheValue;

    if ( Solver != nullptr ) 
    {
      double Value = nlopt_get_stopval( Solver );

      if(std::abs(Value) < HUGE_VAL)
        TheValue = Value;
    }

    return TheValue;
  }

  // One may also stop if the changes in the objective function is less
  // than a given tolerance. The search for a better objective value will
  // stop if the change is less than the tolerance multiplied with the current
  // value of the objective function. Hence this tolerance is relative to the
  // function value. It should be noted that the tolerance value is negative
  // if this criteria is not set, and in that case no value will be returned

  inline void RelativeObjectiveValueTolerance( double Tolerance )
  {
    if ( Solver != nullptr ) 
    {
      nlopt_result Result = nlopt_set_ftol_rel( Solver, Tolerance );
      CheckStatus( Result, "Setting the relative objective value tolerance" );
    } 
    else 
    {
      std::ostringstream ErrorMessage;

      ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
                   << "Trying to set the relative objective function tolerance"
                   << " before the solver has been created";

      throw std::logic_error( ErrorMessage.str() );
    }
  }

  inline std::optional< double > RelativeObjectiveValueTolerance( void )
  {
    std::optional< double > TheValue;

    if ( Solver != nullptr ) 
    {
      double Tolerance = nlopt_get_ftol_rel( Solver );

      if(Tolerance > 0.0)
        TheValue = Tolerance;
    }

    return TheValue;
  }

  // The absolute objective tolerance does not multiply with the value of
  // the objective function, and uses an absolute tolerance in the scale
  // of the objective function. Otherwise similar to the relative tolerance

  inline void AbsoluteObjectiveValueTolerance( double Tolerance )
  {
    if ( Solver != nullptr ) 
    {
      nlopt_result Result = nlopt_set_ftol_abs(Solver, Tolerance);
      CheckStatus(Result, "Setting the absolute objective value tolerance");
    } 
    else 
    {
      std::ostringstream ErrorMessage;

      ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
                   << "Trying to set the absolute objective function tolerance"
                   << " before the solver has been created";

      throw std::logic_error(ErrorMessage.str());
    }
  }

  inline std::optional< double > AbsoluteObjectiveValueTolerance( void )
  {
    std::optional< double > TheValue;

    if ( Solver != nullptr ) 
    {
      double Tolerance = nlopt_get_ftol_abs( Solver );

      if(Tolerance > 0.0)
        TheValue = Tolerance;
    }

    return TheValue;
  }

  // Similarly one may set the tolerance for the change in the variable values
  // relative to the variable value size. The tolerance value is multiplied
  // with each variable value to find if the search can be terminated. The
  // NLopt documentation recommends to also use the absolute variable value
  // tolerance if the solution is close to zero because then the relative
  // tolerance value may vanish.

  inline void RelativeVariableValueTolerance( double Tolerance )
  {
    if ( Solver != nullptr ) 
    {
      nlopt_result Result = nlopt_set_xtol_rel( Solver, Tolerance );
      CheckStatus( Result, "Setting the relative variable value tolerance" );
    } 
    else 
    {
      std::ostringstream ErrorMessage;

      ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
                   << "Trying to set the relative variable value tolerance"
                   << " before the solver has been created";

      throw std::logic_error( ErrorMessage.str() );
    }
  }

  inline std::optional< double > RelativeVariableValueTolerance( void )
  {
    std::optional< double > TheValue;

    if ( Solver != nullptr ) 
    {
      double Tolerance = nlopt_get_xtol_rel( Solver );

      if(Tolerance > 0.0)
        TheValue = Tolerance;
    }

    return TheValue;
  }

  // The absolute variable value tolerance will terminate the search if the 
  // change between two successive iterations is less than the absolute 
  // tolerance value. It should be noted that the tolerance has to be given for 
  // each variable as they may have different scale and therefore different 
  // absolute tolerances. Note that the return value of the function to read 
  // the tolerances returns a vector whose elements are optional tolerance 
  // values. if no solver exists the size of this vector is zero.

  inline void AbsoluteVariableValueTolerance( 
                                     const std::vector< double > & Tolerances )
  {
    if ( Solver != nullptr )
      if ( Tolerances.size() == GetDimension() ) 
      {
        nlopt_result Result = nlopt_set_xtol_abs( Solver, Tolerances.data() );
        CheckStatus( Result, "Setting the absolute variable value tolerance" );
      } 
      else 
      {
        std::ostringstream ErrorMessage;

        ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
                     << "The size of the vector of variable value tolerances "
                     << "was " << Tolerances.size() << " and it must "
                     << "equal the number of variables " << GetDimension();

        throw std::invalid_argument( ErrorMessage.str() );
      }
    else 
    {
      std::ostringstream ErrorMessage;

      ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
                   << "Trying to set the absolute variable value tolerance"
                   << " before the solver has been created";

      throw std::logic_error( ErrorMessage.str() );
    }
  }

  inline std::vector< std::optional< double > > 
  AbsoluteVariableValueTolerance( void )
  {
    std::vector< std::optional< double > > Values;

    if ( Solver != nullptr ) 
    {
      Dimension ProblemSize = GetDimension();
      std::vector< double > Tolerances( ProblemSize, 0.0 );

      nlopt_result Result = nlopt_get_xtol_abs( Solver, Tolerances.data() );
      CheckStatus( Result, "Getting the absolute variable value tolerance" );

      Values.resize( GetDimension() );

      for ( Dimension i = 0; i < ProblemSize; i++ )
        if ( Tolerances[i] > 0.0 )
          Values[i] = Tolerances[i];
    }

    return Values;
  }

  // Even though NLopt provides a utility function to set all the tolerances to
  // the same value, it is better to delegate the first absolute variable value
  // tolerance function because that supports the necessary error checking.

  inline void AbsoluteVariableValueTolerance( double CommonTolerance )
  {
    AbsoluteVariableValueTolerance( std::vector< double >( GetDimension(), 
                                                           CommonTolerance) );
  }

  // It is also possible to stop the search when a maximum number of evaluations
  // of the objective function has been undertaken. It takes an integer value
  // even though it should have been an unsigned value because it must be of
  // the same form as the NLopt library function.

  inline void MaxNumberOfEvaluations( int MaxEval )
  {
    if ( Solver != nullptr ) 
    {
      nlopt_result Result = nlopt_set_maxeval( Solver, MaxEval );
      CheckStatus( Result, "Setting the maximum number of evaluations" );
    } 
    else 
    {
      std::ostringstream ErrorMessage;

      ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
                   << "Trying to set the maximum number of evaluations"
                   << " before the solver has been created";

      throw std::logic_error( ErrorMessage.str() );
    }
  }

  inline std::optional< int > MaxNumberOfEvaluations( void )
  {
    std::optional< int > TheValue;

    if ( Solver != nullptr )
      TheValue = nlopt_get_maxeval( Solver );

    return TheValue;
  }

  // It is also possible to put a time limit on the evaluation and stop when
  // the search time exceeds this value. It is very similar in structure to the
  // limit on number of evaluations. The time is given in seconds whereas it
  // it stored internally in NLopt in a double. Boost numeric cast is used
  // to convert the double to a legal time in seconds.

  inline void MaxTime( std::chrono::seconds Timeout )
  {
    if ( Solver != nullptr ) 
    {
      nlopt_result Result = nlopt_set_maxtime( Solver, 
                            static_cast< double >( Timeout.count() ) );
      CheckStatus( Result, "Setting the evaluation time limit" );
    } 
    else 
    {
      std::ostringstream ErrorMessage;

      ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
                   << "Trying to set the evaluation time out limit"
                   << " before the solver has been created";

      throw std::logic_error( ErrorMessage.str() );
    }
  }

  inline std::optional< std::chrono::seconds > MaxTime( void ) 
  {
    std::optional< std::chrono::seconds > Timeout;

    if ( Solver != nullptr ) 
    {
      double TimeLimit = nlopt_get_maxtime( Solver );

      if ( TimeLimit > 0.0 )
        Timeout = std::chrono::seconds( 
                  boost::numeric_cast< std::chrono::seconds::rep >(TimeLimit) );
    }

    return Timeout;
  }

  // ---------------------------------------------------------------------------
  // Finding a solution
  // ---------------------------------------------------------------------------
  //
  // The solution basically consists of an assignment of the variable values
  // that resulted in the minimal or maximal objective value. The result of
  // the algorithm may also be of interest to see if this is an optimal value
  // conditioned on any of the stop criteria, or even if an error occurred.
  // The function to find a solution should therefore return these three
  // values, and they are structured as an object.

  class OptimalSolution
  {
  public:
  
    const Variables VariableValues;
    const double ObjectiveValue;
    const nlopt_result Status;

    OptimalSolution( const Variables & VariableAssignments, 
                     double ValueOfObjective, nlopt_result SolverStatus )
    : VariableValues( VariableAssignments ), ObjectiveValue( ValueOfObjective ),
      Status( SolverStatus )
    {}

    OptimalSolution(void) = delete;
  };

  // Finally, there is a function to find a solution. It is virtual for the
  // case where an algorithm specific class would need to change its
  // functionality. However, normally the standard behaviour should be
  // sufficient.

  virtual 
  OptimalSolution FindSolution( const Variables & InitialVariableValues )
  {
    // Create the solver if it has not already been done. Note that it uses the
    // default choice of minimization. If the objective should be maximized,
    // the solver should separately be created by a call to the create solver
    // before this function is invoked.

    if ( !Solver )
      CreateSolver( InitialVariableValues.size() );

    // The search for the optimal values starts from the initial variables

    Variables OptimalValues( InitialVariableValues );
    double ObjectiveValue = ObjectiveFunction( InitialVariableValues );

    // The solver is invoked to change these values for the optimal
    // solution

    nlopt_result Result = nlopt_optimize( Solver, OptimalValues.data(), 
                                          &ObjectiveValue );

    // The result can then be returned as an optimal solution leaving the
    // decision about the obtained result to the caller.

    return OptimalSolution( OptimalValues, ObjectiveValue, Result );
  }

  // The constructor is protected as it should only be used by a derived class
  // and it fundamentally initializes the solver pointer to the null pointer.

  OptimizerInterface(void)
  : Optimization::Objective(),
    StatusAction({ { NLOPT_FAILURE, NLOPT_FAILURE }, 
                   { NLOPT_INVALID_ARGS, NLOPT_INVALID_ARGS },
                   { NLOPT_OUT_OF_MEMORY, NLOPT_OUT_OF_MEMORY }, 
                   { NLOPT_ROUNDOFF_LIMITED, NLOPT_ROUNDOFF_LIMITED },
                   { NLOPT_FORCED_STOP, NLOPT_FORCED_STOP }, 
                   { NLOPT_SUCCESS, NLOPT_SUCCESS },
                   { NLOPT_STOPVAL_REACHED, NLOPT_STOPVAL_REACHED }, 
                   { NLOPT_FTOL_REACHED, NLOPT_FTOL_REACHED },
                   { NLOPT_XTOL_REACHED, NLOPT_XTOL_REACHED }, 
                   { NLOPT_MAXEVAL_REACHED, NLOPT_MAXEVAL_REACHED },
                   { NLOPT_MAXTIME_REACHED, NLOPT_MAXTIME_REACHED } }),
    Solver( nullptr )
  {}

  // The destructor allows the correct destruction of all the polymorphic
  // classes and it is therefore public and virtual.

public:

  virtual ~OptimizerInterface( void )
  {
    DeleteSolver();
  }
};

}      // End name space Optimization non-linear
#endif // OPTIMIZATION_NON_LINEAR_OPTIMIZER
