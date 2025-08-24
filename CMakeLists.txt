cmake_minimum_required(VERSION 3.30)
project(Poker VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable CTest/CTest dashboard integration and test discovery
include(CTest)
enable_testing()

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX /arch:AVX2")
  set(CMAKE_EXE_LINKER_FLAGS     "${CMAKE_EXE_LINKER_FLAGS} /LTCG:OFF")
  set(CMAKE_SHARED_LINKER_FLAGS  "${CMAKE_SHARED_LINKER_FLAGS} /LTCG:OFF")
  set(CMAKE_MODULE_LINKER_FLAGS  "${CMAKE_MODULE_LINKER_FLAGS} /LTCG:OFF")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -pedantic -Wno-error=array-bounds -march=native -mavx2")
endif()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

# Conan's CMakeDeps will provide these packages:
find_package(benchmark CONFIG REQUIRED)
find_package(gtest CONFIG REQUIRED)
find_package(bshoshany-thread-pool CONFIG REQUIRED)   # header-only

add_subdirectory(${PROJECT_SOURCE_DIR}/test)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE bshoshany-thread-pool::bshoshany-thread-pool)

add_executable(${PROJECT_NAME}_Game src/game.cpp)
target_link_libraries(${PROJECT_NAME}_Game PRIVATE bshoshany-thread-pool::bshoshany-thread-pool)

add_executable(${PROJECT_NAME}_Benchmark src/benchmarks.cpp)
target_link_libraries(${PROJECT_NAME}_Benchmark PRIVATE benchmark::benchmark benchmark::benchmark_main bshoshany-thread-pool::bshoshany-thread-pool)
