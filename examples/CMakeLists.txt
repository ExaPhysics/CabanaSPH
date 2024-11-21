add_executable(01MovingBlockOfFluid 01_moving_block_of_fluid.cpp)
target_link_libraries(01MovingBlockOfFluid PRIVATE cabanasph)


install(TARGETS 01MovingBlockOfFluid
  DESTINATION ${CMAKE_INSTALL_BINDIR})


add_definitions(-w)
