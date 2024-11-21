#ifndef CABANASPHKernel_HPP
#define CABANASPHKernel_HPP

#include <cmath>


#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

namespace CabanaSPH
{
  KOKKOS_INLINE_FUNCTION
  void compute_quintic_wij(double rij, double h, int dim, double *result){
    double h1 =  1. / h;
    double q =  rij * h1;
    double fac = 1 / M_PI *  7. / 478. * pow(h1, dim);

    double tmp3 = 3. - q;
    double tmp2 = 2. - q;
    double tmp1 = 1. - q;

    double val = 0.;
    if (q > 3.) {
      val = 0.;
    } else if ( q > 2.) {
      val = tmp3 * tmp3 * tmp3 * tmp3 * tmp3;
    } else if ( q > 1.) {
      val = tmp3 * tmp3 * tmp3 * tmp3 * tmp3;
      val -= 6.0 * tmp2 * tmp2 * tmp2 * tmp2 * tmp2;
    } else {
      val = tmp3 * tmp3 * tmp3 * tmp3 * tmp3;
      val -= 6.0 * tmp2 * tmp2 * tmp2 * tmp2 * tmp2;
      val += 15. * tmp1 * tmp1 * tmp1 * tmp1 * tmp1;
    }

    *result = val * fac;
  }


  KOKKOS_INLINE_FUNCTION
  void compute_quintic_gradient_wij(double *xij, double rij, double h, int dim, double *result){
    double h1 =  1. / h;
    double q =  rij * h1;

    double fac = 1. / M_PI *  7. / 478. * pow(h1, dim);

    double tmp3 = 3. - q;
    double tmp2 = 2. - q;
    double tmp1 = 1. - q;

    double val = 0.;
    if (rij > 1e-12){
      if (q > 3.) {
        val = 0.;
      } else if ( q > 2.) {
        val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3;
        val *= h1 / rij;
      } else if ( q > 1.) {
        val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3;
        val += 30.0 * tmp2 * tmp2 * tmp2 * tmp2;
        val *= h1 / rij;
      } else {
        val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3;
        val += 30.0 * tmp2 * tmp2 * tmp2 * tmp2;
        val -= 75.0 * tmp1 * tmp1 * tmp1 * tmp1;
        val *= h1 / rij;
      }
    } else {
      val = 0.;
    }

    double tmp = val * fac;
    result[0] = tmp * xij[0];
    result[1] = tmp * xij[1];
    result[2] = tmp * xij[2];
  }
}

#endif
