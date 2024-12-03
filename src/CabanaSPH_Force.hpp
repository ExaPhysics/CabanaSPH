#ifndef CABANASPHForce_HPP
#define CABANASPHForce_HPP

#include <cmath>

#include <CabanaSPH_Particles.hpp>
#include <CabanaSPH_Kernels.hpp>

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

namespace CabanaSPH
{

  template <class ParticleType>
  void makeArhoZero(ParticleType& particles)
  {
    auto arho = particles.sliceArho();

    Cabana::deep_copy( arho, 0. );
  }

  template <class ParticleType, class NeighListType, class ExecutionSpace>
  void compute_continuity_equation(ParticleType& particles,
                                   const NeighListType& neigh_list,
                                   double dt, int dim)
  {

    auto x = particles.slicePosition();
    auto u = particles.sliceVelocity();
    auto au = particles.sliceAcceleration();
    auto force = particles.sliceForce();
    auto m = particles.sliceMass();
    auto rho = particles.sliceDensity();
    auto p = particles.slicePressure();
    auto h = particles.sliceH();
    auto wij = particles.sliceWij();
    auto arho = particles.sliceArho();

    auto continuity_equation_lambda = KOKKOS_LAMBDA( const int i, const int j )
      {
        /*
          Common to all equations in SPH.

          We compute:
          1.the vector passing from j to i
          2. Distance between the points i and j
          3. Distance square between the points i and j
          4. Velocity vector difference between i and j
          5. Kernel value
          6. Derivative of kernel value
        */
        double pos_i[3] = {x( i, 0 ),
                           x( i, 1 ),
                           x( i, 2 )};

        double pos_j[3] = {x( j, 0 ),
                           x( j, 1 ),
                           x( j, 2 )};

        double pos_ij[3] = {x( i, 0 ) - x( j, 0 ),
                            x( i, 1 ) - x( j, 1 ),
                            x( i, 2 ) - x( j, 2 )};

        double vel_ij[3] = {u( i, 0 ) - u( j, 0 ),
                            u( i, 1 ) - u( j, 1 ),
                            u( i, 2 ) - u( j, 2 )};
        // squared distance
        double r2ij = pos_ij[0] * pos_ij[0] + pos_ij[1] * pos_ij[1] + pos_ij[2] * pos_ij[2];
        // distance between i and j
        double rij = sqrt(r2ij);

        // wij and dwij
        // double wij = 0.;
        double dwij[3] = {0., 0., 0.};

        // h value of particle i
        double h_i = h( i );

        // compute the kernel wij
        // compute_quintic_wij(rij, h_i, &wij);
        // compute the gradient of kernel dwij
        compute_quintic_gradient_wij(pos_ij, rij, h_i, dim, dwij);
        /*
          ====================================
          End: common to all equations in SPH.
          ====================================
        */
        const double mass_j = m( j );
        double vijdotdwij = dwij[0]*vel_ij[0] + dwij[1]*vel_ij[1] + dwij[2]*vel_ij[2];

        arho (i) += mass_j * vijdotdwij;
      };
    Kokkos::RangePolicy<ExecutionSpace> policy(0, x.size());


    Cabana::neighbor_parallel_for( policy,
                                   continuity_equation_lambda,
                                   neigh_list,
                                   Cabana::FirstNeighborsTag(),
                                   Cabana::SerialOpTag(),
                                   "CabanaSPH:Equations:Continuity" );
    Kokkos::fence();
  }


  template <class ParticleType>
  void makeAuZero(ParticleType& particles)
  {
    auto au = particles.sliceAcceleration();

    Cabana::deep_copy( au, 0. );
  }

  template <class ParticleType, class NeighListType, class ExecutionSpace>
  void compute_momentum_equation(ParticleType& particles,
                                 const NeighListType& neigh_list,
                                 double dt,
                                 int dim,
                                 double alpha, double c0)
  {

    auto x = particles.slicePosition();
    auto u = particles.sliceVelocity();
    auto au = particles.sliceAcceleration();
    auto force = particles.sliceForce();
    auto m =particles.sliceMass();
    auto rho = particles.sliceDensity();
    auto p = particles.slicePressure();
    auto h = particles.sliceH();
    auto wij = particles.sliceWij();
    auto arho = particles.sliceArho();

    auto momentum_equation_lambda = KOKKOS_LAMBDA( const int i, const int j )
      {
        /*
          Common to all equations in SPH.

          We compute:
          1.the vector passing from j to i
          2. Distance between the points i and j
          3. Distance square between the points i and j
          4. Velocity vector difference between i and j
          5. Kernel value
          6. Derivative of kernel value
        */
        double pos_i[3] = {x( i, 0 ),
                           x( i, 1 ),
                           x( i, 2 )};

        double pos_j[3] = {x( j, 0 ),
                           x( j, 1 ),
                           x( j, 2 )};

        double pos_ij[3] = {x( i, 0 ) - x( j, 0 ),
                            x( i, 1 ) - x( j, 1 ),
                            x( i, 2 ) - x( j, 2 )};

        double vel_ij[3] = {u( i, 0 ) - u( j, 0 ),
                            u( i, 1 ) - u( j, 1 ),
                            u( i, 2 ) - u( j, 2 )};
        // squared distance
        double r2ij = pos_ij[0] * pos_ij[0] + pos_ij[1] * pos_ij[1] + pos_ij[2] * pos_ij[2];
        // distance between i and j
        double rij = sqrt(r2ij);

        // wij and dwij
        // double wij = 0.;
        double dwij[3] = {0., 0., 0.};

        // h value of particle i
        double h_i = h( i );
        double h_j = h( j );
        double h_ij = (h_i + h_j) / 2.;

        // compute the kernel wij
        // compute_quintic_wij(rij, h_i, &wij);
        // compute the gradient of kernel dwij
        compute_quintic_gradient_wij(pos_ij, rij, h_i, dim, dwij);
        /*
          ====================================
          End: common to all equations in SPH.
          ====================================
        */
        const double mass_i = m( i );
        const double rho_i = rho( i );
        const double p_i = p( i );

        const double mass_j = m( j );
        const double rho_j = rho( j );
        const double p_j = p( j );

        double pij = p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j);
        double tmp = -mass_j * pij;

        // The grad p term
        au (i, 0) += tmp * dwij[0];
        au (i, 1) += tmp * dwij[1];
        au (i, 2) += tmp * dwij[2];

        // ===================================
        // Artificial viscosity force
        // ===================================
        double vijdotrij = vel_ij[0] * pos_ij[0] + vel_ij[1] * pos_ij[1] + vel_ij[2] * pos_ij[2];

        double piij = 0.;
        if (vijdotrij < 0.) {
            double EPS = 0.01 * h_ij * h_ij;
            double rhoij1 = 1 / (rho_i + rho_j);
            double muij = (h_ij * vijdotrij) / (r2ij + EPS);
            piij = -alpha * c0 * muij;
            piij = mass_j * piij * rhoij1;
          }
        au (i, 0) += -piij * dwij[0];
        au (i, 1) += -piij * dwij[1];
        au (i, 2) += -piij * dwij[2];
      };
    Kokkos::RangePolicy<ExecutionSpace> policy(0, x.size());


    Cabana::neighbor_parallel_for( policy,
                                   momentum_equation_lambda,
                                   neigh_list,
                                   Cabana::FirstNeighborsTag(),
                                   Cabana::SerialOpTag(),
                                   "CabanaSPH:Equations:Momentum" );
    Kokkos::fence();
  }
}

#endif
