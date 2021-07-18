#include "GmpUtil.h"
#include <gmp.h>
#include <gmpxx.h>

// ((input - min) * 100) / (max - min)
double CalcPercantage(Int val, Int start, Int range)
{
	//Int val(v);
	val.Add(&start);
	mpz_class x(val.GetBase16().c_str(), 16);
	mpz_class r(range.GetBase16().c_str(), 16);
	x = x - mpz_class(start.GetBase16().c_str(), 16);
	x = x * 100;
	mpf_class y(x);
	y = y / mpf_class(r);
	return y.get_d();
}
