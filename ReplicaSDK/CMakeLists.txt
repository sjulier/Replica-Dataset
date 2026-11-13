# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

set(PTEX_SHADER_DIR "${CMAKE_CURRENT_LIST_DIR}/shaders" CACHE PATH "Where the shaders live")

# The headless renderer creates its GL context through EGL + X11, which only
# exists on Linux (including WSL2). Elsewhere only the viewer is built.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_replica_renderer_default ON)
else()
    set(_replica_renderer_default OFF)
endif()
option(REPLICA_BUILD_RENDERER "Build the headless EGL ReplicaRenderer" ${_replica_renderer_default})

# Replica calls glewInit() itself. Also needed because an installed Pangolin
# only exports its GLEW include path for build-tree consumers.
find_package(GLEW REQUIRED)

find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(STATUS "OpenMP not found: texture/mesh loading will be single-threaded")
endif()

# --- ptex: mesh + texture loading and rendering --------------------------
add_library(ptex SHARED
    ptex/PLYParser.cpp
    ptex/PTexLib.cpp
)
target_include_directories(ptex PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}/include"
)
target_link_libraries(ptex PUBLIC
    Eigen3::Eigen
    GLEW::GLEW
    ${Pangolin_LIBRARIES}
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ptex PRIVATE OpenMP::OpenMP_CXX)
endif()
# STR(SHADER_DIR) in the sources stringifies this, so it must stay unquoted.
target_compile_definitions(ptex PUBLIC SHADER_DIR=${PTEX_SHADER_DIR})
if(NOT MSVC)
    target_compile_options(ptex PRIVATE -Wall)
endif()

# --- ReplicaViewer: interactive Pangolin window --------------------------
add_executable(ReplicaViewer src/viewer.cpp src/MirrorSurface.cpp)
target_link_libraries(ReplicaViewer PRIVATE ptex)

install(TARGETS ptex ReplicaViewer)

# --- ReplicaRenderer: headless rendering via EGL -------------------------
if(REPLICA_BUILD_RENDERER)
    find_package(OpenGL REQUIRED COMPONENTS EGL)
    find_package(X11 REQUIRED)

    add_executable(ReplicaRenderer src/render.cpp src/MirrorSurface.cpp ptex/EGL.cpp)
    target_link_libraries(ReplicaRenderer PRIVATE
        ptex
        OpenGL::EGL
        X11::X11
        ${CMAKE_DL_LIBS}
    )
    install(TARGETS ReplicaRenderer)
else()
    message(STATUS "ReplicaRenderer disabled (needs EGL; Linux only)")
endif()
