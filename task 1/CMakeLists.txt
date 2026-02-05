cmake_minimum_required(VERSION 3.10)
project(TheoryOfParallelism)

# Опция для выбора типа массива (по умолчанию double)
option(USE_FLOAT "Use float instead of double for array type" OFF)

# Установка стандарта C++
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(main main.cpp)

if(USE_FLOAT)
    target_compile_definitions(main PRIVATE USE_FLOAT)
    message(STATUS "Building with float array type")
else()
    message(STATUS "Building with double array type")
endif()