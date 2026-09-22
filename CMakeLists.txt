cmake_minimum_required(VERSION 3.15)
project(kittenml LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(Catch2 3 REQUIRED)

# Header-only interface library for your own kittenml code (include/)
add_library(kittenml INTERFACE)
target_include_directories(kittenml INTERFACE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(kittenml INTERFACE Eigen3::Eigen)

# Demo executable — proves Eigen is wired up correctly
add_executable(demo src/main.cpp)
target_link_libraries(demo PRIVATE kittenml)

# Tests — proves Catch2 is wired up correctly
enable_testing()
add_executable(tests tests/test_basic.cpp tests/test_gradients.cpp)
target_link_libraries(tests PRIVATE kittenml Catch2::Catch2WithMain)
add_test(NAME UnitTests COMMAND tests)
