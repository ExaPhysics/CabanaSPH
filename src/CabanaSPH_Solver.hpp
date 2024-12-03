#ifndef CABANASPHSolver_HPP
#define CABANASPHSolver_HPP

#include <CabanaSPH_Particles.hpp>
#include <CabanaSPH_Force.hpp>
#include <CabanaSPH_Integrator.hpp>

namespace CabanaSPH
{
  template <class MemorySpace, class InputType, class ParticleType>
  class SolverWCSPH
  {
  public:
    using memory_space = MemorySpace;
    using exec_space = typename memory_space::execution_space;

    using particle_type = ParticleType;
    using integrator_type = Integrator<exec_space>;

    // TODO, check this with odler examples
    using neighbor_type =
      Cabana::VerletList<memory_space, Cabana::FullNeighborTag,
                         Cabana::VerletLayout2D, Cabana::SerialOpTag>;
    using neigh_iter_tag = Cabana::SerialOpTag;

    using input_type = InputType;

    SolverWCSPH(input_type _inputs,
                std::shared_ptr<particle_type> _particles,
                double _delta,
                int _dim,
                int _alpha,
                int _c0)
      : inputs( _inputs ),
        particles( _particles ),
        delta( _delta),
        dim( _dim),
        alpha( _alpha),
        c0( _c0)
    {
      num_steps = inputs["num_steps"];
      output_frequency = inputs["output_frequency"];

      // Create integrator.
      dt = inputs["timestep"];
      integrator = std::make_shared<integrator_type>( dt );

      double mesh_min[3] = {
        particles->mesh_lo[0],
        particles->mesh_lo[1],
              particles->mesh_lo[2]};
      double mesh_max[3] = {
        particles->mesh_hi[0],
        particles->mesh_hi[1],
        particles->mesh_hi[2]};
      auto x = particles->slicePosition();
      // This will be changed (No hard coded values)
      auto cell_ratio = 1.0;
      neighbors = std::make_shared<neighbor_type>( x, 0, x.size(),
                                                   delta, cell_ratio,
                                                   mesh_min, mesh_max );
    }

    void run()
    {
      auto x = particles->slicePosition();
      auto cell_ratio = 1.0;
      double mesh_min[3] = {
        particles->mesh_lo[0],
        particles->mesh_lo[1],
        particles->mesh_lo[2]};
      double mesh_max[3] = {
        particles->mesh_hi[0],
        particles->mesh_hi[1],
        particles->mesh_hi[2]};
      // Main timestep loop.
      for ( int step = 0; step <= num_steps; step++ )
        {
          compute_momentum_equation<exec_space>( *particles, *neighbors, dt, dim, alpha, c0 );
          integrator->stage1( *particles );

          compute_continuity_equation( *particles, *neighbors, dt, dim );
          integrator->stage2( *particles );

          // update the neighbours
          neighbors->build( x, 0, x.size(), delta,
                            cell_ratio, mesh_min, mesh_max );

          compute_momentum_equation<exec_space>( *particles, *neighbors, dt, dim, alpha, c0 );
          integrator->stage3( *particles );

          output( step );
        }
    }

    void output( const int step )
    {
      if ( step % output_frequency == 0 )
        {
          std::cout << "We are at " << step << " " << "/ " << num_steps;
          std::cout << std::endl;
          particles->output( step / output_frequency, step * dt);
        }
    }

    int num_steps;
    int output_frequency;
    double dt;

  protected:
    input_type inputs;
    std::shared_ptr<particle_type> particles;
    std::shared_ptr<integrator_type> integrator;
    std::shared_ptr<neighbor_type> neighbors;
    double delta;
    int dim;
    int alpha;
    int c0;
  };


  //---------------------------------------------------------------------------//
  // Creation method.
  template <class MemorySpace, class InputsType, class ParticleType>
  auto createSolverWCSPH(InputsType inputs,
                       std::shared_ptr<ParticleType> particles,
                       double delta)
  {
    return std::make_shared<
      SolverWCSPH<MemorySpace, InputsType, ParticleType>>(inputs, particles, delta);
  }

}

#endif
