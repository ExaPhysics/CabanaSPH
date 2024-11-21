set(HEADERS
  CabanaSPH_Particles.hpp
  CabanaSPH.hpp
  )

set(SOURCES
  CabanaSPH_Particles.cpp
  )

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_library(cabanasph ${SOURCES})

target_link_libraries(cabanasph
  Cabana::Core
  Cabana::Grid
  nlohmann_json::nlohmann_json
  )

target_include_directories(cabanasph
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

install(TARGETS cabanasph LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
