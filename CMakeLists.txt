# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
cmake_minimum_required(VERSION 3.22)

project(Replica LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

# Prefer the pinned dependencies that build.sh installs into 3rdparty/install
# over system packages, so every machine builds against the same versions.
# Point CMAKE_PREFIX_PATH / Eigen3_DIR / Pangolin_DIR elsewhere to override.
set(REPLICA_3RDPARTY_PREFIX "${CMAKE_CURRENT_LIST_DIR}/3rdparty/install"
    CACHE PATH "Prefix holding the pinned third-party dependencies")
list(APPEND CMAKE_PREFIX_PATH "${REPLICA_3RDPARTY_PREFIX}")

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Pangolin 0.9 REQUIRED)
message(STATUS "Eigen3 ${Eigen3_VERSION}: ${Eigen3_DIR}")
message(STATUS "Pangolin ${Pangolin_VERSION}: ${Pangolin_DIR}")

add_subdirectory("./ReplicaSDK")
