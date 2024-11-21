#ifndef CabanaSPH_TIMEINTEGRATOR_HPP
#define CabanaSPH_TIMEINTEGRATOR_HPP

#include <memory>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

namespace CabanaSPH
{
  template <class ExecutionSpace>
  class Integrator
  {
    using exec_space = ExecutionSpace;

    double _dt, _half_dt;
  public:
    Integrator ( double dt )
      : _dt (dt)
    {
      _half_dt = 0.5 * dt;

    }
    ~Integrator() {}

    template <class ParticlesType>
    void stage1(ParticlesType& p){
      auto u = p.sliceVelocity();
      auto au = p.sliceAcceleration();
      auto dt = _dt;
      auto half_dt = _half_dt;
      auto sph_stage1_func = KOKKOS_LAMBDA( const int i )
        {
          u( i, 0 ) += _half_dt * au( i, 0 );
          u( i, 1 ) += _half_dt * au( i, 1 );
          u( i, 2 ) += _half_dt * au( i, 2 );
        };
      Kokkos::RangePolicy<exec_space> policy( 0, u.size() );
      Kokkos::parallel_for( "CabanaSPH::Integrator::Stage1", policy,
                            sph_stage1_func );
    }

    template <class ParticlesType>
    void stage2(ParticlesType& p){
      // _time.start();
      auto x = p.slicePosition();
      auto u = p.sliceVelocity();
      auto arho = p.sliceArho();
      auto dt = _dt;
      auto half_dt = _half_dt;
      auto sph_stage2_func = KOKKOS_LAMBDA( const int i )
        {
          rho( i, 0 ) += dt * arho( i, 0 );

          x( i, 0 ) += dt * u( i, 0 );
          x( i, 1 ) += dt * u( i, 1 );
          x( i, 2 ) += dt * u( i, 2 );
        };
      Kokkos::RangePolicy<exec_space> policy( 0, u.size() );
      Kokkos::parallel_for( "CabanaSPH::Integrator::Stage2", policy,
                            sph_stage2_func );
    }

    template <class ParticlesType>
    void stage3(ParticlesType& p){
      auto u = p.sliceVelocity();
      auto au = p.sliceAcceleration();
      auto dt = _dt;
      auto half_dt = _half_dt;
      auto sph_stage3_func = KOKKOS_LAMBDA( const int i )
        {
          u( i, 0 ) += _half_dt * au( i, 0 );
          u( i, 1 ) += _half_dt * au( i, 1 );
          u( i, 2 ) += _half_dt * au( i, 2 );
        };
      Kokkos::RangePolicy<exec_space> policy( 0, u.size() );
      Kokkos::parallel_for( "CabanaSPH::Integrator::Stage3", policy,
                            sph_stage3_func );
    }

  };
}

#endif // EXAMPM_TIMEINTEGRATOR_HPP
