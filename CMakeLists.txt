cmake_minimum_required(VERSION 4.2)
project(Poker VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable CTest/CTest dashboard integration and test discovery
include(CTest)
enable_testing()

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX /arch:AVX2")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4702")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -pedantic -Wno-error=array-bounds -march=native -mavx2")
endif()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# Conan's CMakeDeps will provide these packages:
find_package(benchmark CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(bshoshany-thread-pool CONFIG REQUIRED)
find_package(Drogon CONFIG REQUIRED)
find_package(glaze CONFIG REQUIRED)
find_package(OpenCLICDLoader CONFIG REQUIRED)
find_package(OpenCLHeadersCpp CONFIG REQUIRED)

# GPU subgame solving: OpenCL plus the kernels, embedded as a generated header.
set(POKER_KERNELS_HEADER ${CMAKE_BINARY_DIR}/generated/subgame_kernels.hpp)
add_custom_command(
  OUTPUT ${POKER_KERNELS_HEADER}
  COMMAND ${CMAKE_COMMAND} -DINPUT=${PROJECT_SOURCE_DIR}/include/cfr/subgame_kernels.cl -DOUTPUT=${POKER_KERNELS_HEADER}
          -DNAME=subgameKernels -P ${PROJECT_SOURCE_DIR}/cmake/embed.cmake
  DEPENDS ${PROJECT_SOURCE_DIR}/include/cfr/subgame_kernels.cl ${PROJECT_SOURCE_DIR}/cmake/embed.cmake
  COMMENT "Embedding subgame_kernels.cl")
add_custom_target(PokerKernels DEPENDS ${POKER_KERNELS_HEADER})
add_library(PokerGpu INTERFACE)
add_dependencies(PokerGpu PokerKernels)
target_include_directories(PokerGpu INTERFACE ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(PokerGpu INTERFACE OpenCL::OpenCL OpenCL::HeadersCpp)

add_subdirectory(${PROJECT_SOURCE_DIR}/test)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE bshoshany-thread-pool::bshoshany-thread-pool)

add_executable(${PROJECT_NAME}_Game src/game.cpp)
target_link_libraries(${PROJECT_NAME}_Game PRIVATE bshoshany-thread-pool::bshoshany-thread-pool)

add_executable(${PROJECT_NAME}_PreflopTable src/generate_preflop_table.cpp)
target_link_libraries(${PROJECT_NAME}_PreflopTable PRIVATE bshoshany-thread-pool::bshoshany-thread-pool)

add_executable(${PROJECT_NAME}_Abstraction src/generate_abstraction.cpp)
target_link_libraries(${PROJECT_NAME}_Abstraction PRIVATE bshoshany-thread-pool::bshoshany-thread-pool)

add_executable(${PROJECT_NAME}_Train src/train.cpp)
target_link_libraries(${PROJECT_NAME}_Train PRIVATE bshoshany-thread-pool::bshoshany-thread-pool)

add_executable(${PROJECT_NAME}_PushFold src/push_fold.cpp)
target_link_libraries(${PROJECT_NAME}_PushFold PRIVATE bshoshany-thread-pool::bshoshany-thread-pool)

add_executable(${PROJECT_NAME}_Lbr src/lbr.cpp)
target_link_libraries(${PROJECT_NAME}_Lbr PRIVATE bshoshany-thread-pool::bshoshany-thread-pool PokerGpu)

add_executable(${PROJECT_NAME}_Benchmark src/benchmarks.cpp)
target_link_libraries(${PROJECT_NAME}_Benchmark PRIVATE benchmark::benchmark benchmark::benchmark_main bshoshany-thread-pool::bshoshany-thread-pool)

add_executable(${PROJECT_NAME}_Api src/api.cpp)
target_link_libraries(${PROJECT_NAME}_Api PRIVATE Drogon::Drogon bshoshany-thread-pool::bshoshany-thread-pool glaze::glaze)
add_executable(${PROJECT_NAME}_Slumbot src/slumbot.cpp)
target_link_libraries(${PROJECT_NAME}_Slumbot PRIVATE Drogon::Drogon glaze::glaze bshoshany-thread-pool::bshoshany-thread-pool PokerGpu)
