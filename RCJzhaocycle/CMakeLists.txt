cmake_minimum_required(VERSION 3.16)

project(RCJzhaocycle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_library(rcj_arc_detector
    src/arc_detector.cpp
)

target_include_directories(rcj_arc_detector
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(rcj_arc_detector
    PUBLIC
        ${OpenCV_LIBS}
)

target_compile_options(rcj_arc_detector
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -O3>
)

add_executable(arc_benchmark
    tools/arc_benchmark.cpp
)

target_link_libraries(arc_benchmark
    PRIVATE
        rcj_arc_detector
)

target_compile_options(arc_benchmark
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -O3>
)
