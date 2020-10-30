/*==============================================================================
Interpolation

There are many situations where one would like to represent a function based
on sampled data points. However, most optimization algorithms assume that the
functions to be optimized are continuous. The solution is then to represent
the function as an interpolation of the sample data points.

This file provides a homogeneous interface to several interpolation algorithms
from the GNU Scientific Library (GSL) [1] and Boost::Math [2]. Currently only
univariate interpolation has been defined.

The sampled data points can be given through a pre-defined container, either a
map or as some kind of linear structure with an iterator, or it can be given
in a comma separated file (CSV) assuming that each row contains <x, f(x)>
value pairs. Ben Strasser's fast C++ CSV Reader class [3] is used for parsing
the file.

Following the principle of using the newest C++ standard, this implementation
requires a compiler that is capable of handling C++20, in particular the
constexpr virtual functions.

References:
[1] https://www.gnu.org/software/gsl/
[2] https://www.boost.org/doc/libs/1_72_0/libs/math/doc/html/interpolation.html
[3] https://github.com/ben-strasser/fast-cpp-csv-parser
[4] M. Steffen: "A Simple Method for Monotonic Interpolation in One Dimension",
    Astronomy & Astrophysics, Vol. 239, No. 1-2, pp. 443-450, November 1990
[4] Jerrold Fried and Stanley Zietz: "Curve fitting by spline and Akima
    methods: possibility of interpolation error and its suppression",
    Physics in Medicine and Biology, Vol. 18, No. 4, July 1973
[5] Hiroshi Akima: "A New Method of Interpolation and Smooth Curve Fitting
    Based on Local Procedures", Journal of the ACM, Vol. 17, No. 4, pp.
    589–602, October 1970
[5] Michael S. Floater and Kai Hormann: "Barycentric rational interpolation
    with no poles and high rates of approximation", Numerische Mathematik,
		Vol. 107, pp. 315–331, 2007
[6] Claus Schneider and Wilhelm Werner: "Some New Aspects of Rational
    Interpolation", Mathematics of Computation, Vol 47, No. 175, pp. 285-299,
		July 1986
[7] Giesela Engeln-Müllges and Frank Uhlig: Numerical Algorithms with C,
    Springer-Verlag Berlin Heidelberg, ISBN 978-3-642-61074-5, 1996


Author and Copyright: Geir Horn, University of Oslo 2020
License: European Union Public Licence 1.2 (EUPL-1.2)
         https://ec.europa.eu/info/european-union-public-licence_en

==============================================================================*/

#ifndef OPTIMIZATION_INTERPOLATION
#define OPTIMIZATION_INTERPOLATION

#include <string>                             // Strings
#include <sstream>                            // For error messages
#include <stdexcept>                          // For standard exceptions
#include <type_traits>                        // Type checking
#include <iterator>                           // Iterating over data
#include <map>                                // Storing sample data
#include <vector>                             // C-style vectors for GSL
#include <filesystem>                         // File names
#include <mutex>                              // Protecting GSL accelerator
#include <optional>                           // f(x) when defined or not
#include <algorithm>                          // STL algorithms
#include <numeric>                            // Numerical STL algorithms
#include <limits>                             // Numeric limits of types

#include <gsl/gsl_interp.h>                   // GSL algorithms
#include <gsl/gsl_errno.h>                    // GSL error reporting
#include <boost/range/adaptor/map.hpp>        // Range adaptors
#include <boost/numeric/conversion/cast.hpp>  // Safe numeric casts
#include <boost/math/interpolators/barycentric_rational.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>

#include "csv.h"                              // The CSV parser

namespace Interpolation
{

/*==============================================================================

 Naming algorithms

==============================================================================*/

class Algorithm
{
private:

  // In order to ensure the strict typing of algorithms, they are all
  // defined in a flat scoped enumeration list. U means uniformly sampled
	// abscissa whereas NU means non-uniformly sampled abscissa. The algorithms
	// are given in preference order so that non-uniform algorithms are generally
	// preferred over uniformly sampled, and so that the periodic algorithms are
	// the least preferred.

	enum class Type
	{
		Steffen,           // GSL   NU: Monotonous piecewise cubic
		Akima,             // GSL   NU: Natural endpoints
		Barycentric,       // Boost NU: O(n) evaluation time
		Spline,            // GSL   NU: Piecewise cubic
		Linear,            // GSL   NU: Simple
		Polynomial,        // GSL   NU: Easily introduces oscillations
		BSpline,           // Boost  U: cubic version is used
		WhittakerShannon,  // Boost  U: compactly supported functions
		PeriodicSpline,    // GSL   NU: Cubic spline with periodic boundaries
		PeriodicAkima,     // GSL   NU: Periodic endpoints
		Fourier            // Boost  U: Periodic function interpolation
	};

public:

	// The next declaration is a trick to give other classes the possibility
	// to define arguments and return values of the enumeration type but forcing
	// the use of the below constants to actually set these values since the
	// enumeration itself is private.

	using InterpolationType = Type;

	// For error reporting it is useful to have a textual string representation
	// of the various algorithm types, and there is a simple function to produce
	// the required text for a given type.

	static constexpr const char * Description( Type Method )
	{
		switch( Method )
		{
			case Type::Steffen:
				return "Steffen's method";
				break;
			case Type::Akima:
				return "Akima spline";
				break;
			case Type::Barycentric:
				return "Barycentric rational";
				break;
			case Type::Spline:
				return "Cubic spline";
				break;
			case Type::Linear:
				return "Linear interpolation";
				break;
			case Type::Polynomial:
				return "Polynomial interpolation";
				break;
			case Type::BSpline:
				return "B-Spline";
				break;
			case Type::WhittakerShannon:
				return "Whittaker-Shannon method";
				break;
			case Type::PeriodicSpline:
				return "Periodic spline";
				break;
			case Type::PeriodicAkima:
				return "Periodic Akima spline";
				break;
			case Type::Fourier:
				return "Periodic cardinal trigonometric interpolation";
				break;
		}

		return "Unknown method";
	}

	// The actual definitions are made in terms of the types of algorithms
	// to improve readability of the subsequent implementations. Some algorithms
	// requiring an uniformly sampled abscissa, i.e. the function values are only
	// known at equally spaced points. For these algorithms, it is necessary to
	// check this condition prior to constructing the interpolation class. To
	// be able to inherit the interpolation class, or for interpolation methods
	// implemented without a default constructor, there is a small class to
	// check this condition, and it will throw an invalid argument exception if
	// the condition of a uniformly sampled abscissa is not met.

	struct Univariate
	{
		static constexpr Type
			Linear      = Type::Linear,
			Polynomial  = Type::Polynomial,
			Spline      = Type::Spline,
			Akima       = Type::Akima,
			Steffen     = Type::Steffen,
			Barycentric = Type::Barycentric;

		struct Periodic
		{
			static constexpr Type
			  Akima   = Type::PeriodicAkima,
			  Spline  = Type::PeriodicSpline;
		};

		struct Uniform
		{
			static constexpr Type
			  BSpline          = Type::BSpline,
				WhittakerShannon = Type::WhittakerShannon;

			struct Periodic
			{
				static constexpr Type
				  Fourier = Type::Fourier;
			};

			// The test class will first create a vector of differences between the
			// abscissa values and then check if all of these are equal. If not then
			// the invalid argument exception error will be thrown. Note that the
			// per element test is done against the precision of the real type to
			// avoid a test for exact zero.

			class AbscissaTest
			{
			public:
				template< class AbscissaIterator >
				AbscissaTest( AbscissaIterator xBegin, AbscissaIterator xEnd,
											Type Method )
				{
					using RealType =
					typename std::iterator_traits< AbscissaIterator >::value_type;

					static_assert( std::is_arithmetic< RealType >::value,
												 "The abscissa type must be a numeric type!"	);

					std::vector< RealType	> Delta;
					std::adjacent_difference( xBegin, xEnd, std::back_inserter( Delta ) );

					if( !std::all_of( Delta.begin(), Delta.end(),
							 [&](RealType delta)->bool{ return
							 ( std::max( delta, Delta.front() )
							    - std::min( delta, Delta.front() ) ) <
									  2 * std::numeric_limits< RealType >::epsilon(); } ) )
					{
						std::ostringstream ErrorMessage;

						ErrorMessage << __FILE__ << " at line " << __LINE__ << ": "
						             << "The abscissa values are not equally spaced "
												 << "points as required by the interpolation method"
												 << Description( Method );

					  throw std::invalid_argument( ErrorMessage.str() );
					}
				}
			};
		};

		};

	// Later it is possible to add interpolations in multiple dimensions here

	// ---------------------------------------------------------------------------
	// Comparators
	// ---------------------------------------------------------------------------
	//
	// The function to chose one out of two types will select the one with the
  // least value based on the ordering of the different interpolation types
	// in the above enumeration.

	static constexpr InterpolationType Choose( InterpolationType A,
																						 InterpolationType B )
	{
		if ( static_cast< unsigned int >(A) <= static_cast< unsigned int >(B) )
			return A;
	  else
			return B;
	}

	// There are two functions to test if a given algorithm is uniformly sampled
	// or if it is periodic

	static constexpr bool UniformQ( InterpolationType A )
	{
		switch( A )
	  {
			case Univariate::Uniform::BSpline:
			case Univariate::Uniform::WhittakerShannon:
			case Univariate::Uniform::Periodic::Fourier:
				return true;
			default:
				return false;
		}
	}

	static constexpr bool PeriodicQ( InterpolationType A )
	{
		switch( A )
	  {
			case Univariate::Periodic::Akima:
			case Univariate::Periodic::Spline:
			case Univariate::Uniform::Periodic::Fourier:
				return true;
			default:
				return false;
		}
	}
};

/*==============================================================================

 Generic interface

==============================================================================*/
//
// The generic interface is the base class of all the interpolation functions
// and it basically provides the map of the values of the function being
// interpolated by the various methods and constructors to initialize this map.
// The class is abstract and should never be used on its own.

template< class AbscissaType, class OrdinateType = AbscissaType >
class GenericFunction
{
public:

	// The types are defined so that derived classes knows the double types.

	using Abscissa = AbscissaType;
	using Ordinate = OrdinateType;

private:

	// First a check that the given types are arithmetic, i.e. integer or
	// real numbers

	static_assert( std::is_arithmetic< AbscissaType >::value &&
								 std::is_arithmetic< OrdinateType >::value,
								 "Interpolation data types must be arithmetic!"	);

	// If the given types are acceptable, then the vectors holding separately
	// the data are defined. They must be kept separate because some algorithms
	// may require the data for every evaluation of the interpolated function.
	// Note that it is assumed that the abscissa data are sorted in ascending
	// order.

	std::vector< AbscissaType > AbscissaData;
	std::vector< OrdinateType > OrdinateData;

	// In order to ensure that the abscissa is sorted, the data is assumed to
	// be initialized from a map. The basic initialization function is therefore
	// based on two map iterators. It is a template since the value stored in
	// the map can be arithmetic type.

	template< class MapIterator >
	void StoreMapData( MapIterator DataPoint, MapIterator End )
	{
		using MapAbcissaType =
			typename std::iterator_traits< MapIterator >::value_type::first_type;
		using MapOrdinateType =
			typename std::iterator_traits< MapIterator >::value_type::second_type;

		static_assert( std::is_arithmetic< MapAbcissaType  >::value &&
									 std::is_arithmetic< MapOrdinateType >::value,
									 "The map must contain numerical values" );

		// Types are numeric so we can use the safe numeric cast of boost to
		// fill the values into the sample data map.

		for( ; DataPoint != End; ++DataPoint )
		{
			AbscissaData.push_back(
				boost::numeric_cast< AbscissaType >( DataPoint->first ) );
			OrdinateData.push_back(
				boost::numeric_cast< OrdinateType >( DataPoint->second ) );
		}

		// At least two data points must be given, otherwise an invalid argument
		// exception will be thrown

		if( AbscissaData.size() < 2 )
		{
			std::ostringstream ErrorMessage;

	    ErrorMessage <<  __FILE__ << " at line " << __LINE__ << ": "
			  				   << "The given data set  does not contain sufficient data";

	    throw std::invalid_argument( ErrorMessage.str() );
		}
	}

protected:

	// There are functions to generate a range that can be used to iterate over
	// the abscissa and ordinate values separately. The objects returned from
	// these functions has a 'begin' and 'end' functions returning iterators to
	// go over the data.

  inline const std::vector< AbscissaType > & AbscissaValues( void )
	{ return AbscissaData; }

	inline const std::vector< OrdinateType > & OrdinateValues( void )
	{ return OrdinateData; }

public:

	// Often one would be satisfied by just knowing the number of samples. This
	// is returned as the size variable supported by the map, which should be
	// an integral type, usually size_t, but it is implementation dependent.

	inline typename std::vector< AbscissaType >::size_type	SampleSize( void )
	{ return AbscissaData.size(); }

	// External users can only get constant iterators to the sample data

	inline const typename std::vector< AbscissaType >::const_iterator
	AbscissaBegin( void )
	{ return AbscissaData.cbegin(); }

	inline const typename std::vector< AbscissaType >::const_iterator
	AbscissaEnd( void )
	{ return AbscissaData.cend(); }

	inline const typename std::vector< OrdinateType >::const_iterator
	OrdinateBegin( void )
	{ return OrdinateData.cbegin(); }

	inline const typename std::vector< OrdinateType >::const_iterator
	OrdinateEnd( void )
	{ return OrdinateData.cbegin(); }

	// ---------------------------------------------------------------------------
	// Constructors
	// ---------------------------------------------------------------------------
	//
  // The easiest part is when the data is already available in a map, and when
	// the 'begin' and 'end' iterators are given. It must be checked that the
	// data type of the input is numerical, and the data will then be copied
	// to the sampled data

	template< class MapIterator >
	GenericFunction( MapIterator DataPoint, MapIterator End )
	{ StoreMapData( DataPoint, End );	}

	// If the map is given directly, the handling can just be forwarded to
	// the storage function.

	template< typename XValueType, typename YValueType,
	          class Comparator, class Allocator >
  GenericFunction(
  const std::map< XValueType, YValueType, Comparator, Allocator > & DataPoints )
	{ StoreMapData( DataPoints.cbegin(), DataPoints.cend() ); }

	// It is also possible that the abscissa values and the ordinate values are
	// already prepared in the separate containers. In this case the 'begin' and
	// 'end' should be given for the abscissa values, and it is implicitly assumed
	// that the ordinate values are at least as many. In this case it is
	// necessary to first store the data in a map to ensure that the abscissa is
	// correctly sorted, and then use this map to initialize the data vectors.

	template< class AbscissaIterator, class OrdinateIterator >
	GenericFunction( AbscissaIterator XValues, AbscissaIterator XValuesEnd,
									 OrdinateIterator YValues )
	{
		std::map< typename std::iterator_traits< AbscissaIterator >::value_type,
					    typename std::iterator_traits< OrdinateIterator >::value_type >
					    GivenData;

		for( ; XValues != XValuesEnd; ++XValues, ++YValues )
			GivenData.emplace(*XValues, *YValues );

		StoreMapData( GivenData.cbegin(), GivenData.cend() );
	}

	// The last constructors will read the data from a Comma Separated File (CSV)
	// assuming that there is one data point <x,y> per line (record).

	GenericFunction( const std::filesystem::path & FileName )
	{
		// First two variable place holders are defined as the CSV library uses a
		// C-style interface filling two variables with data.

		AbscissaType X;
		OrdinateType Y;

		// Parse two columns from the file, using space as separator and ignore only
    // tabs.

	  io::CSVReader<2, io::trim_chars<'\t'>, io::no_quote_escape<' '> >
	      CSVParser( FileName );

		// The file is not supposed to contain a header and so it is defined to
		// create an exception if already defined.

		CSVParser.set_header("X", "Y");

		// Then the file can be parsed and the content converted and stored in a
		// temporary map that is finally stored to the data vectors.

		std::map< AbscissaType, OrdinateType > GivenData;

		while ( CSVParser.read_row( X, Y ) )
			GivenData.emplace( X, Y );

		StoreMapData( GivenData.cbegin(), GivenData.cend() );
	}

	// There are copy constructors and move constructors basically passing the
	// stored data on to the sample data map directly.

	GenericFunction(
		Interpolation::GenericFunction< AbscissaType, OrdinateType > && Other )
	: AbscissaData( Other.AbscissaData ), OrdinateData( Other.OrdinateData )
  {};

	GenericFunction(
		const Interpolation::GenericFunction< AbscissaType, OrdinateType > & Other )
	: AbscissaData( Other.AbscissaData ), OrdinateData( Other.OrdinateData )
	{ };

	// The default constructor is not possible as the class must have the data

	GenericFunction( void ) = delete;

	// ---------------------------------------------------------------------------
	// Algorithm function
	// ---------------------------------------------------------------------------
	//
	// There is a function for the classes in the hierarchy to inspect what kind
	// of algorithm is used for the interpolation involved. It is declared
	// constexpr so that it can be used at compile time to check the algorithm
	// avoiding a separate type definition in the various method classes. Note
	// that this is a C++20 feature and the compiler should be able to support
	// this.

	constexpr virtual Algorithm::InterpolationType Method( void ) const = 0;

	// ---------------------------------------------------------------------------
	// Value functions
	// ---------------------------------------------------------------------------
	//
	// Evaluating the function for a given argument depends on the interpolation
	// method and the functions are therefore just defined. The value version
	// will not throw, but return an optional that will not have a value if the
	// argument was out of bounds for the interpolating method. The operator
	// version will check the optional value returned from Value, and throw
	// a invalid argument exception if the value function does not return a value.

	virtual std::optional< OrdinateType > Value( AbscissaType Argument ) = 0;

	OrdinateType operator() ( AbscissaType Argument )
	{
		std::optional< OrdinateType > Result( Value( Argument ) );

		if( Result ) return Result.value();
		else
		{
			std::ostringstream ErrorMessage;

	    ErrorMessage <<  __FILE__ << " at line " << __LINE__ << ": "
			  				   << "The argument " << Argument << " is not valid for "
									 << "the interpolation defined on the closed interval ["
									 << DomainLower() << ", " << DomainUpper() << "] for the "
									 << " interpolation method "
									 << Algorithm::Description( Method() );

	    throw std::invalid_argument( ErrorMessage.str() );
		}
	}

	// ---------------------------------------------------------------------------
	// Domain functions
	// ---------------------------------------------------------------------------
	//

	inline AbscissaType DomainLower (void) const
	{ return AbscissaData.front(); }

	inline AbscissaType DomainUpper (void) const
	{ return AbscissaData.back(); }

	inline bool DomainQ (AbscissaType x) const
	{ return (DomainLower() <= x) && (x <= DomainUpper()); }

	// ---------------------------------------------------------------------------
	// Derivatives and integrals
	// ---------------------------------------------------------------------------
	//
	// There are functions to find the derivatives of the interpolated function
	// at certain points. However it is understood that not all derivatives
	// exists for all algorithms, and the default behaviour is therefore to
	// return an empty value

	virtual
	std::optional< OrdinateType > FirstDerivative( AbscissaType Argument )
	{ return std::optional< OrdinateType >();	}

	virtual
	std::optional< OrdinateType > SecondDerivative( AbscissaType Argument )
	{ return std::optional< OrdinateType >(); }

	virtual
	std::optional< OrdinateType > Integral( AbscissaType From, AbscissaType To )
  { return std::optional< OrdinateType >(); }

	// ---------------------------------------------------------------------------
	// Destructor
	// ---------------------------------------------------------------------------
	//
  // The destructor does nothing, but it is a virtual place holder to ensure
	// that the destructor of a derived class is properly invoked.

	virtual ~GenericFunction()
	{ }
};

/*==============================================================================

 GNU Scientific Library base

==============================================================================*/
//
// GSL is C-style object oriented meaning that a data structure will be
// allocated and then populated with algorithm specific parameters. This is
// common behaviour to all GSL interpolation algorithms and it is therefore
// managed by a common base class.
//
// This class also inherits the Generic Function object specialized for double
// precision real valued data since the GSL algorithms are hard coded for
// doubles. The generic function is inherited as virtual mainly to force the
// algorithm specific classes to explicitly call its constructor to ensure it
// initializes the sample data map before the GSL interpolation object is
// allocated as the size of the data store is needed.

class GSL : public GenericFunction< double >
{
private:

	// ---------------------------------------------------------------------------
	// Interpolation object and accelerator
	// ---------------------------------------------------------------------------
	//
	// The parameter storage for GSL is done in a object that is dynamically
	// allocated depending on the algorithm type to be used.

	gsl_interp * InterpolationObject = nullptr;

  // The GSL needs an accelerator object holding the state of searches and an
  // interpolation object holding the static state (coefficients) computed from
  // the data. The accelerator object contains information about the
	// interpolation function that may speed up some computations.

	gsl_interp_accel * AcceleratorObject = nullptr;

	// There is a function to allocate the two GSL objects and initialize them
	// from the data stored in the Generic Function object. It should be noted
	// that when computing the parameters of the interpolation algorithm, C-style
	// arrays are needed and the above access functions are constructed for this
	// purpose.

	void AllocateAndCompute( const gsl_interp_type * GSLAlgorithm )
	{
		InterpolationAlgorithm = GSLAlgorithm;

		// Allocate the GSL data storage

		InterpolationObject = gsl_interp_alloc( GSLAlgorithm, SampleSize() );
		AcceleratorObject   = gsl_interp_accel_alloc();

		// Compute the interpolation object data

		gsl_interp_init( InterpolationObject,
										 AbscissaValues().data(), OrdinateValues().data(),
									   SampleSize() );
	}

	// Various functions using the accelerator object may actually write to it,
  // and therefore it must be protected in order to allow the same
  // interpolation object to be used from concurrent threads. The interpolation
  // structure is passed as a constant to every function, so parallelism
  // should not be a problem for this object.

  std::mutex AcceleratorLock;

	// It stores the GSL algorithm type. The only thing this is useful for is
	// to be able to manipulate interpolation objects with arithmetic operations
	// and to copy one object to another.

	const gsl_interp_type * InterpolationAlgorithm;

	// ---------------------------------------------------------------------------
	// Error handling
	// ---------------------------------------------------------------------------
	//
	// In the same way the error code returned from some of the value producing
	// functions will be stored so that when an empty optional is received it
	// is possible to check the error value.

  int GSLErrorCode;

public:

	// This code can be reported as a string if it should be communicated to the
	// user somehow.

	inline bool ErrorQ( void )
	{ return GSLErrorCode != GSL_SUCCESS; }

	inline std::string ErrorMessage( void )
	{ return gsl_strerror( GSLErrorCode ); }

	// ---------------------------------------------------------------------------
	// Constructors
	// ---------------------------------------------------------------------------
	//
	// The same set of constructors as for the Generic Function is supported,
	// and they all take an extra parameter being the GSL ID of the algorithm.
	// This is used to allocate the GSL data structures correctly.
	//
	// The first constructor is used when iterators for a map of data is
	// directly given. The underlying numerical type can be whatever as the
	// conversion to doubles will be managed by the generic function.

	template< class MapIterator >
	GSL( MapIterator FirstDataPoint, MapIterator LastDataPoint,
			 const gsl_interp_type * GSLAlgorithm	)
	: GenericFunction( FirstDataPoint, LastDataPoint )
	{ AllocateAndCompute( GSLAlgorithm );	}

	// The second constructor assumes that the map is given, and this requires
	// template arguments for all possible template arguments for the map.

	template< typename XValueType, typename YValueType,
	          class Comparator, class Allocator >
	GSL( const std::map< XValueType, YValueType, Comparator, Allocator >
						 & DataPoints, const gsl_interp_type * GSLAlgorithm )
  : GenericFunction( DataPoints )
	{ AllocateAndCompute( GSLAlgorithm );}

	// The third constructor takes data already filled in separate sequence
	// containers for the abscissa values and the ordinate values. As long as
	// the data can be converted to a double it is OK.

	template< class AbscissaIterator, class OrdinateIterator >
	GSL( AbscissaIterator XValues, AbscissaIterator XValuesEnd,
			 OrdinateIterator YValues, const gsl_interp_type * GSLAlgorithm )
	: GenericFunction( XValues, XValuesEnd, YValues )
	{ AllocateAndCompute( GSLAlgorithm ); }

	// The last constructor is taking a CSV file of data and parse this
	// to get the sample data

	GSL( const std::filesystem::path & FileName,
			 const gsl_interp_type * GSLAlgorithm )
	: GenericFunction( FileName )
	{ AllocateAndCompute( GSLAlgorithm );}

	// Moving and copying is allowed. Moving means taking over the initialized
	// objects from the other type, whereas copying means basically recomputing
	// based on the data from the other.

	GSL( GSL && Other )
	: GenericFunction( Other )
	{
		std::lock_guard< std::mutex > Lock( Other.AcceleratorLock );

		InterpolationObject = Other.InterpolationObject;
		AcceleratorObject   = Other.AcceleratorObject;

		Other.InterpolationObject = nullptr;
		Other.AcceleratorObject   = nullptr;
	}

	GSL( const GSL & Other )
	: GenericFunction( Other )
	{ AllocateAndCompute( Other.InterpolationAlgorithm );	}

	// The standard constructor is disallowed

	GSL( void ) = delete;

	// ---------------------------------------------------------------------------
	// Evaluating the interpolated function
	// ---------------------------------------------------------------------------
	//
	// The evaluation function of the GSL library is used, and if it returns a
	// domain error, it means that the argument is outside of the bounds of the
	// abscissa. In this case an empty optional will be returned. It seems that
	// this could be readily tested by checking the domain and not try to do an
	// evaluation, but for periodic interpolations one may be able to use the
	// interpolation for extrapolation and as such the decision is left to the
	// GSL routines.

	virtual std::optional< double > Value( double Argument )
	{
		std::lock_guard< std::mutex > Lock( AcceleratorLock );

		double InterpolatedValue;

		GSLErrorCode = gsl_interp_eval_e( InterpolationObject,
											AbscissaValues().data(), OrdinateValues().data(),
										  Argument, AcceleratorObject, &InterpolatedValue );

		if ( GSLErrorCode == GSL_EDOM )
		  return std::optional< double >();
		else
		  return std::optional< double >( InterpolatedValue );
	}

	// ---------------------------------------------------------------------------
	// Derivatives and integrals
	// ---------------------------------------------------------------------------
	//
	// There are functions to find the derivatives of the interpolated function
	// at certain points. If the argument is outside of the domain for the given
	// operation, an empty optional is returned. Again, the decision on what to
	// return is made by the GSL.

	virtual
	std::optional< double > FirstDerivative( double Argument )
	{
		std::lock_guard< std::mutex > Lock( AcceleratorLock );

		double InterpolatedValue;

		GSLErrorCode = gsl_interp_eval_deriv_e( InterpolationObject,
										  AbscissaValues().data(), OrdinateValues().data(),
											Argument, AcceleratorObject, &InterpolatedValue  );

		if ( ErrorQ() )
		  return std::optional< double >();
		else
		  return std::optional< double >( InterpolatedValue );
	}

	virtual
	std::optional< double > SecondDerivative( double Argument )
	{
		std::lock_guard< std::mutex > Lock( AcceleratorLock );

		double InterpolatedValue;

		GSLErrorCode = gsl_interp_eval_deriv2_e( InterpolationObject,
										  AbscissaValues().data(), OrdinateValues().data(),
											Argument, AcceleratorObject, &InterpolatedValue  );

		if ( ErrorQ() )
		  return std::optional< double >();
		else
		  return std::optional< double >( InterpolatedValue );
	}

	virtual
	std::optional< double > Integral( double From, double To )
  {
		std::lock_guard< std::mutex > Lock( AcceleratorLock );

		double InterpolatedValue;

		GSLErrorCode = gsl_interp_eval_integ_e( InterpolationObject,
										  AbscissaValues().data(), OrdinateValues().data(),
											From, To, AcceleratorObject, &InterpolatedValue  );

		if ( ErrorQ() )
		  return std::optional< double >();
		else
		  return std::optional< double >( InterpolatedValue );
	}

	// ---------------------------------------------------------------------------
	// Destructor
	// ---------------------------------------------------------------------------
	//
  // The destructor ensures that the allocated objects are properly destroyed
	// if they have been allocated and if they are still under the ownership of
	// this object.

	virtual ~GSL()
	{
	  if ( InterpolationObject != nullptr )
	    gsl_interp_free( InterpolationObject );

	  if ( AcceleratorObject != nullptr )
	  {
	    std::lock_guard< std::mutex > Lock( AcceleratorLock );
	    gsl_interp_accel_free( AcceleratorObject );
	  }
	}
};

/*==============================================================================

 Interpolating functions

==============================================================================*/
//
// An interpolating function is a template specialized for each of the available
// algorithms. Fundamentally, each class needs to specify the constructors and
// the virtual functions, and the implementation of the actual calculations
// will be managed by the defined base classes. The references are at the top
// of this file. The non-uniform algorithms are defined first in alphabetical
// order, and then the uniform algorithms and finally the periodic
// interpolations
//
// The function templates for specifying the real types used can be understood
// as iterators for some algorithms

template< Algorithm::InterpolationType TheAlgorithm,
         class AbscissaReal = double, class OrdinateReal = AbscissaReal >
class Function;

// ---------------------------------------------------------------------------
// Steffen's method
// ---------------------------------------------------------------------------
//
// Steffen's method [4] guarantees monotonicity in the interpolation and has
// therefore been chosen as the default method here, with the Akima spline
// as a good candidate to consider if additivity is desired. Because of the
// monotonicity Steffen's method guarantees that minima and maxima can only
// occur exactly at the data points, and there can never be spurious
// oscillations between data points. The interpolated function is piecewise
// cubic in each interval. The resulting curve and its first derivative are
// guaranteed to be continuous, but the second derivative may be discontinuous.

template<>
class Function< Algorithm::Univariate::Steffen >
: public GSL
{
public:

	// Then the constructors can be defined.

	template< class MapIterator >
	Function( MapIterator FirstDataPoint, MapIterator LastDataPoint)
	: GSL( FirstDataPoint, LastDataPoint, gsl_interp_steffen )
	{}

	template< typename XValueType, typename YValueType,
	          class Comparator, class Allocator >
	Function( const std::map< XValueType, YValueType, Comparator, Allocator >
						& DataPoints )
	: GSL( DataPoints, gsl_interp_steffen )
	{}

	template< class AbscissaIterator, class OrdinateIterator >
	Function( AbscissaIterator XValues, AbscissaIterator XValuesEnd,
			      OrdinateIterator YValues )
	: GSL( XValues, XValuesEnd, YValues, gsl_interp_steffen )
	{}

  Function( const std::filesystem::path & FileName )
	: GSL( FileName, gsl_interp_steffen )
	{}

	Function( Function< Algorithm::Univariate::Steffen > && Other )
	: GSL( Other )
	{}

	Function( Function< Algorithm::Univariate::Steffen > & Other )
	: GSL( Other )
	{}

	constexpr virtual Algorithm::InterpolationType Method( void ) const
	{ return Algorithm::Univariate::Steffen; }

	virtual ~Function()
	{}
};

// ---------------------------------------------------------------------------
// Akima
// ---------------------------------------------------------------------------
//
// Splines generally fit cubic polynomials to each pair of points along the
// function, matching the first and second order derivatives in the given
// data points. However, these splines can oscillate near outliers [5], and a
// more robust alternative would be to use Akima splines [6].

template<>
class Function< Algorithm::Univariate::Akima >
: public GSL
{
public:

	// Then the constructors can be defined.

	template< class MapIterator >
	Function( MapIterator FirstDataPoint, MapIterator LastDataPoint)
	: GSL( FirstDataPoint, LastDataPoint, gsl_interp_akima )
	{}

	template< typename XValueType, typename YValueType,
	          class Comparator, class Allocator >
	Function( const std::map< XValueType, YValueType, Comparator, Allocator >
						& DataPoints )
	: GSL( DataPoints, gsl_interp_akima )
	{}

	template< class AbscissaIterator, class OrdinateIterator >
	Function( AbscissaIterator XValues, AbscissaIterator XValuesEnd,
			      OrdinateIterator YValues )
	: GSL( XValues, XValuesEnd, YValues, gsl_interp_akima )
	{}

  Function( const std::filesystem::path & FileName )
	: GSL( FileName, gsl_interp_akima )
	{}

	Function( Function< Algorithm::Univariate::Akima > && Other )
	: GSL( Other )
	{}

	Function( Function< Algorithm::Univariate::Akima > & Other )
	: GSL( Other )
	{}

	constexpr virtual Algorithm::InterpolationType Method( void ) const
	{ return Algorithm::Univariate::Akima; }

	virtual ~Function()
	{}
};

// ---------------------------------------------------------------------------
// Barycentric Rational Interpolation
// ---------------------------------------------------------------------------
//
// Barycentric rational interpolation is a high-accuracy interpolation method
// for non-uniformly spaced samples. It requires 𝑶(N) time for construction,
// and 𝑶(N) time for each evaluation. The algorithm is based on [6] and the
// derivatives in [7]. There is an optional parameter to all constructors
// giving the interpolation order. The default value for the order of the
// approximation is 3, and hence the accuracy is 𝑶(h^4) where h is the step
// size if the data is sampled at regular intervals. In general, for an order d
// then the error is 𝑶(h^(d+1)). This may cause additional complexity in the
// calculations, but may be needed in special cases. The interpolation order
// is stored in a read-only data field.

template< class AbscissaReal >
class Function< Algorithm::Univariate::Barycentric, AbscissaReal >
: public GenericFunction< AbscissaReal >
{
private:

	// Shorthand notation for the template base class

	using DataStore = GenericFunction< AbscissaReal >;

	// The class has the Boost interpolation class

	boost::math::barycentric_rational< AbscissaReal > Barycentric;

public:

  // The type definitions of the generic function is used

	using typename DataStore::Abscissa;
	using typename DataStore::Ordinate;

 	// The interpolation order is stored to be able to copy the interpolation
	// object. It is a constant so it can be read by other classes but never
  // changed after construction. It is expected as a size_t parameter by the
	// Barycentric interpolation class, and it is consequently declared as size_t.

	const std::size_t InterpolationOrder;

  // Then the constructors are defined

	template< class MapIterator >
	Function( MapIterator FirstDataPoint, MapIterator LastDataPoint,
						std::size_t Order = 3 )
	: DataStore( FirstDataPoint, LastDataPoint ),
	  Barycentric( DataStore::AbscissaBegin(), DataStore::AbscissaEnd(),
								 DataStore::OrdinateBegin(), Order ),
	  InterpolationOrder( Order )
	{}

	template< typename XValueType, typename YValueType,
	          class Comparator, class Allocator >
	Function( const std::map< XValueType, YValueType, Comparator, Allocator >
						& DataPoints, std::size_t Order = 3 )
	: DataStore( DataPoints ),
	  Barycentric( DataStore::AbscissaBegin(), DataStore::AbscissaEnd(),
								 DataStore::OrdinateBegin(), Order ),
	  InterpolationOrder( Order )
	{}

	template< class AbscissaIterator, class OrdinateIterator >
	Function( AbscissaIterator XValues, AbscissaIterator XValuesEnd,
			      OrdinateIterator YValues, std::size_t Order = 3 )
	: DataStore( XValues, XValuesEnd, YValues ),
	  Barycentric( DataStore::AbscissaBegin(), DataStore::AbscissaEnd(),
								 DataStore::OrdinateBegin(), Order ),
	  InterpolationOrder( Order )
	{}

	Function( const std::filesystem::path & FileName, std::size_t Order = 3 )
	: DataStore( FileName ),
	  Barycentric( DataStore::AbscissaBegin(), DataStore::AbscissaEnd(),
								 DataStore::OrdinateBegin(), Order ),
	  InterpolationOrder( Order )
	{}

	template< class OtherReal >
	Function( Function< Algorithm::Univariate::Barycentric, OtherReal > && Other )
	: DataStore( Other.AbscissaBegin(), Other.AbscissaEnd(),
							 Other.OrdinateBegin() ),
	  Barycentric( DataStore::AbscissaBegin(), DataStore::AbscissaEnd(),
								 DataStore::OrdinateBegin(), Other.InterpolationOrder ),
	  InterpolationOrder( Other.InterpolationOrder )
	{}

	template< class OtherReal >
	Function( Function< Algorithm::Univariate::Barycentric, OtherReal > & Other )
	: DataStore( Other.AbscissaBegin(), Other.AbscissaEnd(),
							 Other.OrdinateBegin() ),
	  Barycentric( DataStore::AbscissaBegin(), DataStore::AbscissaEnd(),
								 DataStore::OrdinateBegin(), Other.InterpolationOrder ),
	  InterpolationOrder( Other.InterpolationOrder )
	{}

	Function( void ) = delete;

	// The value is obtained through the value function, that will return an
	// unassigned optional if the requested argument is outside of the domain
	// of the interpolation

	virtual std::optional< Ordinate >	Value( Abscissa Argument )
	{
		if( DataStore::DomainQ( Argument) )
			return std::optional< Ordinate >(	Barycentric( Argument)	);
		else
			return std::optional< Ordinate >();
	}

	// Only the first derivative is defined directly by the boost class.

	virtual	std::optional< Ordinate > FirstDerivative( Abscissa Argument )
	{
		if( DataStore::DomainQ( Argument ) )
			return std::optional< Ordinate >( Barycentric.prime( Argument ) );
		else
			return std::optional< Ordinate >();
	}

	// The second derivative is not directly given, and it must be computed by
	// some numerical approximation. TODO

	// The integration is made by the Gauss-Kronrod quadrature as this method
	// is also the preferred method for the GNU Scientific library. The adaptive
	// version from Boost is used with standard parameters.

	virtual	std::optional< Ordinate > Integral( Abscissa From, Abscissa To )
  {
		return std::optional< Ordinate >(
	    boost::math::quadrature::gauss_kronrod< Abscissa, 15 >::integrate(
		    [&](Abscissa x){ return Barycentric(x); },
				std::max( From, DataStore::DomainLower() ),
			  std::min( To, DataStore::DomainUpper() )  )
		);
	}

	// Finally, the function returning the algorithm used.

	constexpr virtual Algorithm::InterpolationType Method( void ) const
	{ return Algorithm::Univariate::Barycentric; }

	// The destructor does nothing that is not managed automatically

	virtual ~Function()
	{}
};

// ---------------------------------------------------------------------------
// Spline
// ---------------------------------------------------------------------------
//
// The cubic spline with natural boundary conditions provided by the GSL is
// used here. The resulting curve is a piecewise cubic polynomial on each
// interval, with matching first and second derivatives at the supplied
// data-points. The second derivative is chosen to be zero at the first point
// and last point.

template<>
class Function< Algorithm::Univariate::Spline >
: public GSL
{
public:

	// Then the constructors can be defined.

	template< class MapIterator >
	Function( MapIterator FirstDataPoint, MapIterator LastDataPoint)
	: GSL( FirstDataPoint, LastDataPoint, gsl_interp_cspline )
	{}

	template< typename XValueType, typename YValueType,
	          class Comparator, class Allocator >
	Function( const std::map< XValueType, YValueType, Comparator, Allocator >
						& DataPoints )
	: GSL( DataPoints, gsl_interp_cspline )
	{}

	template< class AbscissaIterator, class OrdinateIterator >
	Function( AbscissaIterator XValues, AbscissaIterator XValuesEnd,
			      OrdinateIterator YValues )
	: GSL( XValues, XValuesEnd, YValues, gsl_interp_cspline )
	{}

  Function( const std::filesystem::path & FileName )
	: GSL( FileName, gsl_interp_cspline )
	{}

	Function( Function< Algorithm::Univariate::Spline > && Other )
	: GSL( Other )
	{}

	Function( Function< Algorithm::Univariate::Spline > & Other )
	: GSL( Other )
	{}

	constexpr virtual Algorithm::InterpolationType Method( void ) const
	{ return Algorithm::Univariate::Spline; }

	virtual ~Function()
	{}
};

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------
//
// Linear interpolation fits a straight line between successive pairs of
// the data points, i.e. it fits a first order polynomial to each pair.
// The first derivative is therefore, in general, discontinuous at the data
// points. It is fast and memory efficient, but may not capture the sampled
// function well.

template<>
class Function< Algorithm::Univariate::Linear >
: public GSL
{
public:

	// Then the constructors can be defined.

	template< class MapIterator >
	Function( MapIterator FirstDataPoint, MapIterator LastDataPoint)
	: GSL( FirstDataPoint, LastDataPoint, gsl_interp_linear )
	{}

	template< typename XValueType, typename YValueType,
	          class Comparator, class Allocator >
	Function( const std::map< XValueType, YValueType, Comparator, Allocator >
						& DataPoints )
	: GSL( DataPoints, gsl_interp_linear )
	{}

	template< class AbscissaIterator, class OrdinateIterator >
	Function( AbscissaIterator XValues, AbscissaIterator XValuesEnd,
			      OrdinateIterator YValues )
	: GSL( XValues, XValuesEnd, YValues, gsl_interp_linear )
	{}

  Function( const std::filesystem::path & FileName )
	: GSL( FileName, gsl_interp_linear )
	{}

	Function( Function< Algorithm::Univariate::Linear > && Other )
	: GSL( Other )
	{}

	Function( Function< Algorithm::Univariate::Linear > & Other )
	: GSL( Other )
	{}

	constexpr virtual Algorithm::InterpolationType Method( void ) const
	{ return Algorithm::Univariate::Linear; }

	virtual ~Function()
	{}
};

// ---------------------------------------------------------------------------
// Polynomial
// ---------------------------------------------------------------------------
//
// This method should only be used for interpolating small numbers of points
// because polynomial interpolation introduces large oscillations, even for
// well-behaved datasets. The number of terms in the interpolating polynomial
// is equal to the number of points.

template<>
class Function< Algorithm::Univariate::Polynomial >
: public GSL
{
public:

	// Then the constructors can be defined.

	template< class MapIterator >
	Function( MapIterator FirstDataPoint, MapIterator LastDataPoint)
	: GSL( FirstDataPoint, LastDataPoint, gsl_interp_polynomial )
	{}

	template< typename XValueType, typename YValueType,
	          class Comparator, class Allocator >
	Function( const std::map< XValueType, YValueType, Comparator, Allocator >
						& DataPoints )
	: GSL( DataPoints, gsl_interp_polynomial )
	{}

	template< class AbscissaIterator, class OrdinateIterator >
	Function( AbscissaIterator XValues, AbscissaIterator XValuesEnd,
			      OrdinateIterator YValues )
	: GSL( XValues, XValuesEnd, YValues, gsl_interp_polynomial )
	{}

  Function( const std::filesystem::path & FileName )
	: GSL( FileName, gsl_interp_polynomial )
	{}

	Function( Function< Algorithm::Univariate::Polynomial > && Other )
	: GSL( Other )
	{}

	Function( Function< Algorithm::Univariate::Polynomial > & Other )
	: GSL( Other )
	{}

	constexpr virtual Algorithm::InterpolationType Method( void ) const
	{ return Algorithm::Univariate::Polynomial; }

	virtual ~Function()
	{}
};


// ---------------------------------------------------------------------------
// Akima periodic
// ---------------------------------------------------------------------------
//
// Non-rounded Akima spline with periodic boundary conditions. This algorithm
// uses the non-rounded corner algorithm of Wodicka. It seems difficult to find
// a better reference than [x] for this method.


/*==============================================================================

 Combining functions

==============================================================================*/
//
// Combining two interpolating functions is not obvious. There are two things
// to consider:
// 1. If the two abscissae do not overlap, what should be used for the ordinate
//    values between the two ranges?
// 2. What should be the resulting algorithm and data types of the combined
//    function if the two interpolating functions are interpolated with
//    different algorithms?
// For the first question, one could consider using a linear extrapolation
// binding the two functions. For now, this will simply be a gap in the
// abscissa values, and how to deal with this depends on the interpolation
// algorithm used for the resulting function.
//
// The second question is more involved as one may select one of the two
// algorithms used by the two functions combined, or one may select a completely
// different algorithm. It is easy if the two algorithms of the two functions
// are equal, then the output should be of the same kind. If the two algorithms
// of the two functions are different, it is preferable to chose the one of
// the two algorithms supporting unequal sampling of the abscissa. This will
// also allow preserving the original abscissa for this function since it may
// already be unequally sampled. If both functions are based on equally sampled
// ranges, then the new abscissa will also be equally sampled. In all cases,
// the defined priority of the algorithms will be used, see the algorithm class
// above. These decisions are captured by the above test functions for the
// interpolation algorithms provided above.
//
// The ordinal values of the two involved functions are computed by the value
// function that provides an optional to the combination function. The
// combination function returns an optional. If this has no value, the point
// will be dropped in case of a non-uniform abscissa, and it will be taken as
// zero in the case of a uniform abscissa.

template< Algorithm::InterpolationType FirstAlgorithm,
          Algorithm::InterpolationType SecondAlgorithm,
					class BinaryOperation >
auto CombineFunctions ( Function< FirstAlgorithm  > f1,
                        Function< SecondAlgorithm > f2,
												BinaryOperation Combine  )
{
	// Finding the new algorithm and the types for the abscissa and the ordinate

	constexpr Algorithm::InterpolationType
	NewAlgorithm( Algorithm::Choose( FirstAlgorithm, SecondAlgorithm ) );

  using AbscissaType = typename Function< NewAlgorithm >::Abscissa;
	using OrdinateType = typename Function< NewAlgorithm >::Ordinate;

	// The real point is to construct a new set of function values by using the
	// combination operator over an abscissa that is either explicitly or
	// implicitly given depending on whether the function is uniform or not.

	if constexpr ( Algorithm::UniformQ( NewAlgorithm ) )
  {
		// In the uniform case the sample step size of the involved algorithms
		// will be estimated, an then new ordinates will be generated by using
		// the smallest step size for the entire combined domain. The actual
		// abscissa is generated by the constructor of the interpolating function.

		AbscissaType Delta =
									std::min( ( f1.DomainUpper() - f1.DomainLower() )/f1.Size(),
													  ( f2.DomainUpper() - f2.DomainLower() )/f2.Size() ),
						     DomainLower = std::min( f1.DomainLower(), f2.DomainLower() ),
						     DomainUpper = std::max( f1.DomainUpper(), f2.DomainUpper() );

  	std::vector< OrdinateType > CombinedOrdinate;

		for( AbscissaType x  = DomainLower; x <= DomainUpper; x += Delta	)
		   CombinedOrdinate.push_back(
			   Combine( f1.Value( x ), f2.Value( x ) ).value_or( 0 ) );

	  return Function< NewAlgorithm >(
		  CombinedOrdinate.begin(), CombinedOrdinate.end(), DomainLower, Delta );
	}
	else
  {
		// The abscissa for a non uniform combination is found by taking the
		// union of the abscissae of the two involved function. Then the new
		// function values are found by applying the combination function to the
		// two functions evaluated for the data points of the combined abscissa

  	std::vector< AbscissaType >	CombinedAbscissa;

	  std::set_union( f1.AbscissaBegin(), f1.AbscissaEnd(),
										f2.AbscissaBegin(), f2.AbscissaEnd(),
										std::back_inserter( CombinedAbscissa ) );

		// The combined data are stored in a map to allow for missing data if the
		// combination cannot be evaluated for some abscissa values.

		std::map< AbscissaType, OrdinateType > SampleData;

		for( AbscissaType x : CombinedAbscissa )
		{
			std::optional< OrdinateType > f( Combine( f1.Value(x), f2.Value(x) ) );
			if( f ) SampleData.emplace( x, f.value() );
		}

		return Function< NewAlgorithm  >( SampleData );
	}
}

} // Name space Interpolation

/*==============================================================================

 Arithmetic operations

==============================================================================*/
//
// Defining the arithmetic operators is now easily done using lambdas on the
// above combination function after first finding out what the return type
// for the ordinate will be.
//
// This is relatively simple for addition

template< Interpolation::Algorithm::InterpolationType FirstAlgorithm,
          Interpolation::Algorithm::InterpolationType SecondAlgorithm >
auto operator+ ( Interpolation::Function< FirstAlgorithm  > f1,
								 Interpolation::Function< SecondAlgorithm > f2 )
{
	using OrdinateType =
	typename Interpolation::Function< Interpolation::Algorithm::Choose(
													FirstAlgorithm, SecondAlgorithm ) >::Ordinate;

	return Interpolation::CombineFunctions( f1, f2,
				 []( const std::optional< OrdinateType > & y1,
						 const std::optional< OrdinateType > & y2 )
				 { return std::optional< OrdinateType >(
									  y1.value_or(0) + y2.value_or(0) ); } );
}

// Subtraction follows exactly the same pattern

template< Interpolation::Algorithm::InterpolationType FirstAlgorithm,
          Interpolation::Algorithm::InterpolationType SecondAlgorithm >
auto operator- ( Interpolation::Function< FirstAlgorithm  > f1,
								 Interpolation::Function< SecondAlgorithm > f2 )
{
	using OrdinateType =
	typename Interpolation::Function< Interpolation::Algorithm::Choose(
													FirstAlgorithm, SecondAlgorithm ) >::Ordinate;

	return Interpolation::CombineFunctions( f1, f2,
				 []( const std::optional< OrdinateType > & y1,
						 const std::optional< OrdinateType > & y2 )
				 { return std::optional< OrdinateType >(
									  y1.value_or(0) - y2.value_or(0) ); } );
}

// Multiplication is also trivial since the combined function will be zero
// where the two domains do not overlap.

template< Interpolation::Algorithm::InterpolationType FirstAlgorithm,
          Interpolation::Algorithm::InterpolationType SecondAlgorithm >
auto operator* ( Interpolation::Function< FirstAlgorithm  > f1,
								 Interpolation::Function< SecondAlgorithm > f2 )
{
	using OrdinateType =
	typename Interpolation::Function< Interpolation::Algorithm::Choose(
													FirstAlgorithm, SecondAlgorithm ) >::Ordinate;

	return Interpolation::CombineFunctions( f1, f2,
				 []( const std::optional< OrdinateType > & y1,
						 const std::optional< OrdinateType > & y2 )
				 { return std::optional< OrdinateType >(
									  y1.value_or(0) * y2.value_or(0) ); } );
}

// Division is more complicated since the divisor function is per definition
// unknown outside its domain, and by default assumed to be zero. It makes
// no sense to return not-a-number (NaN) when the divisor is undefined or zero,
// and in this case an empty optional will be returned

template< Interpolation::Algorithm::InterpolationType FirstAlgorithm,
          Interpolation::Algorithm::InterpolationType SecondAlgorithm >
auto operator/ ( Interpolation::Function< FirstAlgorithm  > f1,
								 Interpolation::Function< SecondAlgorithm > f2 )
{
	using OrdinateType =
	typename Interpolation::Function< Interpolation::Algorithm::Choose(
													FirstAlgorithm, SecondAlgorithm ) >::Ordinate;

	return Interpolation::CombineFunctions( f1, f2,
				 []( const std::optional< OrdinateType > & y1,
						 const std::optional< OrdinateType > & y2 )
				 { if( y2 && ( y2.value() != 0 ) )
					  return std::optional< OrdinateType >( y1.value_or(0) / y2.value() );
					 else
						return std::optional< OrdinateType >();
				 } );
}

#endif
