#ifndef CabanaSPHParticles_HPP
#define CabanaSPHParticles_HPP

#include <memory>
#include <filesystem> // or #include <filesystem> for C++17 and up

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>


namespace fs = std::filesystem;


#define DIM 3

namespace CabanaSPH
{
  template <class MemorySpace, int Dimension, int MaxCnts, int MaxNoWalls>
  class Particles
  {
  public:
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;
    static constexpr int dim = Dimension;

    using double_type = Cabana::MemberTypes<double>;
    using int_type = Cabana::MemberTypes<int>;
    using vec_double_type = Cabana::MemberTypes<double[dim]>;
    using vec_int_type = Cabana::MemberTypes<int[dim]>;

    // FIXME: add vector length.
    // FIXME: enable variable aosoa.
    using aosoa_double_type = Cabana::AoSoA<double_type, memory_space, 1>;
    using aosoa_int_type = Cabana::AoSoA<int_type, memory_space, 1>;
    using aosoa_vec_double_type = Cabana::AoSoA<vec_double_type, memory_space, 1>;
    using aosoa_vec_int_type = Cabana::AoSoA<vec_int_type, memory_space, 1>;

    std::array<double, DIM> mesh_lo;
    std::array<double, DIM> mesh_hi;

    // Constructor which initializes particles on regular grid.
    template <class ExecSpace>
    Particles( const ExecSpace& exec_space, std::size_t no_of_particles, std::string output_folder_name )
    {
      _no_of_particles = no_of_particles;
      _output_folder_name = output_folder_name;
      // create the output folder if it doesn't exist

      if (!fs::is_directory(_output_folder_name) || !fs::exists(_output_folder_name)) { // Check if src folder exists
        fs::create_directory(_output_folder_name); // create src folder
      }

      resize( _no_of_particles );
      createParticles( exec_space );
      // Set dummy values here, reset them in particular examples
      for ( int d = 0; d < dim; d++ )
        {
          mesh_lo[d] = 0.0;
          mesh_hi[d] = 0.0;
        }
    }

    // Constructor which initializes particles on regular grid.
    template <class ExecSpace>
    void createParticles( const ExecSpace& exec_space )
    {
      auto x = slicePosition();

      auto create_particles_func = KOKKOS_LAMBDA( const int i )
        {
          for (int j=0; j < DIM; j++){
            // x( i, j ) = DIM * i + j;
          }
        };
      Kokkos::RangePolicy<ExecSpace> policy( 0, x.size() );
      Kokkos::parallel_for( "create_particles_lambda", policy,
                            create_particles_func );
    }

    template <class ExecSpace, class FunctorType>
    void updateParticles( const ExecSpace, const FunctorType init_functor )
    {
      Kokkos::RangePolicy<ExecSpace> policy( 0, _no_of_particles );
      Kokkos::parallel_for(
                           "CabanaPD::Particles::update_particles", policy,
                           KOKKOS_LAMBDA( const int pid ) { init_functor( pid ); } );
    }

    auto slicePosition()
    {
      return Cabana::slice<0>( _x, "positions" );
    }
    auto slicePosition() const
    {
      return Cabana::slice<0>( _x, "positions" );
    }

    auto sliceVelocity()
    {
      return Cabana::slice<0>( _u, "velocities" );
    }
    auto sliceVelocity() const
    {
      return Cabana::slice<0>( _u, "velocities" );
    }

    auto sliceAcceleration()
    {
      return Cabana::slice<0>( _au, "accelerations" );
    }
    auto sliceAcceleration() const
    {
      return Cabana::slice<0>( _au, "accelerations" );
    }

    auto sliceForce()
    {
      return Cabana::slice<0>( _force, "forces" );
    }
    auto sliceForce() const
    {
      return Cabana::slice<0>( _force, "forces" );
    }

    auto sliceMass() {
      return Cabana::slice<0>( _m, "mass" );
    }
    auto sliceMass() const
    {
      return Cabana::slice<0>( _m, "mass" );
    }

    auto sliceDensity() {
      return Cabana::slice<0>( _rho, "density" );
    }
    auto sliceDensity() const
    {
      return Cabana::slice<0>( _rho, "density" );
    }

    auto slicePressure() {
      return Cabana::slice<0>( _p, "pressure" );
    }
    auto slicePressure() const
    {
      return Cabana::slice<0>( _p, "pressure" );
    }

    auto sliceH() {
      return Cabana::slice<0>( _h, "smoothing_length" );
    }
    auto sliceH() const
    {
      return Cabana::slice<0>( _h, "smoothing_length" );
    }

    auto sliceWij() {
      return Cabana::slice<0>( _wij, "wij" );
    }
    auto sliceWij() const
    {
      return Cabana::slice<0>( _wij, "wij" );
    }

    auto sliceArho() {
      return Cabana::slice<0>( _arho, "arho" );
    }
    auto sliceArho() const
    {
      return Cabana::slice<0>( _arho, "arho" );
    }

    void resize(const std::size_t n)
    {
      _no_of_particles = n;
      _x.resize( n );
      _u.resize( n );
      _au.resize( n );
      _force.resize( n );
      _m.resize( n );
      _rho.resize( n );
      _p.resize( n );
      _h.resize( n );
      _wij.resize( n );
      _arho.resize( n );
    }

    void output(  const int output_step,
                  const double output_time,
                  const bool use_reference = true )
    {
      // _output_timer.start();

#ifdef Cabana_ENABLE_HDF5
      Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
                                                              h5_config,
                                                              _output_folder_name+"/particles",
                                                              // "particles",
                                                              MPI_COMM_WORLD,
                                                              output_step,
                                                              output_time,
                                                              _no_of_particles,
                                                              slicePosition(),
                                                              sliceVelocity(),
                                                              sliceAcceleration(),
                                                              sliceForce(),
                                                              sliceMass(),
                                                              sliceDensity(),
                                                              slicePressure(),
                                                              sliceH(),
                                                              sliceWij(),
                                                              sliceArho());
      // #else
      // #ifdef Cabana_ENABLE_SILO
      //       Cabana::Grid::Experimental::SiloParticleOutput::
      //        writePartialRangeTimeStep(
      //                                  "particles", output_step, output_time,
      //                                  _no_of_particles,
      //                                  slicePosition(),
      //                                  sliceVelocity(),
      //                                  sliceAcceleration(),
      //                                  sliceMass(),
      //                                  sliceDensity(),
      //                                  sliceRadius());
#else
      std::cout << "No particle output enabled.";
      // log( std::cout, "No particle output enabled." );
      // #endif
#endif

      // _output_timer.stop();
    }

  private:
    int _no_of_particles;
    aosoa_vec_double_type _x;
    aosoa_vec_double_type _u;
    aosoa_vec_double_type _au;
    aosoa_vec_double_type _force;
    aosoa_double_type _m;
    aosoa_double_type _rho;
    aosoa_double_type _p;
    aosoa_double_type _h;
    aosoa_double_type _wij;
    aosoa_double_type _arho;
    std::string _output_folder_name;

#ifdef Cabana_ENABLE_HDF5
    Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
#endif

    // Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
  };

} // namespace CabanaSPH

#endif
