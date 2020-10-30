/*==============================================================================
Interpolation test

The purpose of this file is to show how to use the interpolation algorithms
and unit test the implementation. It uses the third example of GSL to illustrate
the idea. It implements the data set taken from [1] to show the difference
between different interpolation algorithms. It generates two data files,
"SampleData" and "Interpolation". The latter datafile has the abscissa values
in the first column and then one column for each of the interpolation methods
tested.

It is necessary to include paths to the Interpolation header and the CSV header
it uses, and link with the GSL library that uses the CBLAS library and support
for C++20, e.g.

g++ InterpolationTest.cpp -o InterpolationTest -I.. -I../../CSV -std=c++2a -lgsl -lcblas

References:
[1] James M. Hyman: "Accurate Monotonicity Preserving Cubic Interpolation",
    SIAM Journal on Scientific and Statistical Computing, Volume 4, Issue 4,
		pp. 645–654, 1983

Author and Copyright: Geir Horn, 2020
License: LGPL 3.0 (GPL if GSL algorithms are used)
==============================================================================*/

#include <array>
#include <fstream>

#include "Interpolation.hpp"

int main(int argc, char **argv)
{
	// Defining the data set

	std::array< double, 9>
		x{ 7.99, 8.09, 8.19, 8.7, 9.2, 10.0, 12.0, 15.0, 20.0 },
		y{ 0.0, 2.76429e-5, 4.37498e-2, 0.169183, 0.469428, 0.943740, 0.998636,
			 0.999919, 0.999994 };

  // Writing these data points to a file

  std::ofstream SampleData( "SampleData.dta");

	for( auto Abscissa = x.begin(), Ordinate = y.begin(); Abscissa != x.end();
			 ++Abscissa, ++Ordinate )
		SampleData << *Abscissa << " " << *Ordinate << std::endl;

	SampleData.close();

  // Defining the interpolation functions for these data. It is interesting to
	// observe that the Barycentric rational misses the values completely.

  Interpolation::Function< Interpolation::Algorithm::Univariate::Akima >
  AkimaSpline( x.begin(), x.end(), y.begin() );

	Interpolation::Function< Interpolation::Algorithm::Univariate::Barycentric >
	BarycentricRational( x.begin(), x.end(), y.begin() );

	Interpolation::Function< Interpolation::Algorithm::Univariate::Steffen >
	Steffen( x.begin(), x.end(), y.begin() );

	Interpolation::Function< Interpolation::Algorithm::Univariate::Spline >
	Spline( x.begin(), x.end(), y.begin() );

  // Calculate some interpolated values over the whole range of the data.
	// The ranges are conveniently taken from the Akima spline class. There is
	// an interesting point in that one could think that it will be possible to
	// compute the step size and then add this repeatedly to x[0]. However, this
	// will make the end point slightly larger than the upper bound of the
	// domain owing to rounding errors. Increasing the precision to long
	// double makes the iteration work correctly

  std::ofstream InterpolatedData( "Interpolated.dta");

  long double StepSize =
		          ( AkimaSpline.DomainUpper() - AkimaSpline.DomainLower() ) / 100.0;

  for( long double Abscissa = AkimaSpline.DomainLower();
			             Abscissa < AkimaSpline.DomainUpper() + StepSize;
							     Abscissa += StepSize )
		InterpolatedData << Abscissa << " "
										 << AkimaSpline( Abscissa ) << " "
										 << BarycentricRational( Abscissa ) << " "
										 << Steffen( Abscissa ) << " "
										 << Spline( Abscissa )
										 << std::endl;

  // Close the file and return.

  InterpolatedData.close();

	return EXIT_SUCCESS;
}

