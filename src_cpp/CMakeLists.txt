cmake_minimum_required(VERSION 3.5)
project(kumozu)


set(BLA_VENDOR "Intel10_64lp")


find_package( BLAS REQUIRED )
message( STATUS BLAS found: ${BLAS_LIBRARIES} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fopenmp -std=c++1y")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_GLIBCXX_DEBUG -DKUMOZU_DEBUG")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

#include_directories(/opt/intel/mkl/include)

file( GLOB LIB_SOURCES lib/*.cpp )
file( GLOB LIB_HEADERS lib/*.h )
add_library( kumozu ${LIB_SOURCES} ${LIB_HEADERS} )

include_directories(lib)

# The public executable
file(GLOB APP_SOURCE_FILES "app/*.cpp")
add_executable(main ${APP_SOURCE_FILES})

# The experimental executable
# COMMENT THE NEXT TWO LINES OUT IN PUBLIC VERSION!!!
#file(GLOB EXPERIMENTAL_APP_SOURCE_FILES "app_experimental/*.cpp")
#add_executable(main_experimental ${EXPERIMENTAL_APP_SOURCE_FILES})

find_package(Boost 1.58.0 COMPONENTS system iostreams filesystem REQUIRED)

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/")
include_directories (${Boost_INCLUDE_DIRS} ${MKL_INCLUDE_DIR})

target_link_libraries(main kumozu ${Boost_LIBRARIES} ${BLAS_LIBRARIES})

# COMMENT THE FOLLOWING LINE IN PUBLIC VERSION!!!
#target_link_libraries(main_experimental kumozu ${Boost_LIBRARIES} ${BLAS_LIBRARIES}) 
