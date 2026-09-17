# XCCL - unified NCCL/RCCL collective-communication wrapper.
# M2 skeleton. Builds with a plain C compiler on the host; nothing in the
# build depends on nccl.h / rccl.h / CUDA / ROCm headers.

cmake_minimum_required(VERSION 3.10)

project(xccl VERSION 0.1.0 LANGUAGES C)

option(XCCL_BUILD_TESTS               "Build XCCL unit tests (fake backends)" ON)
option(XCCL_BUILD_TESTS_INTEGRATION   "Build real-backend integration test (requires NCCL/RCCL at run time)" OFF)
option(XCCL_BUILD_EXAMPLES            "Build example programs" ON)

set(XCCL_SOURCES
    src/xcc_error.c
    src/xcc_loader.c
    src/xcc_vtable.c
    src/xcc_api.c
    src/backends/nccl.c
    src/backends/rccl.c)

if(WIN32)
    list(APPEND XCCL_SOURCES src/xcc_platform_windows.c)
else()
    list(APPEND XCCL_SOURCES src/xcc_platform_posix.c)
endif()

add_library(xcc ${XCCL_SOURCES})
add_library(xcc::xcc ALIAS xcc)

target_include_directories(xcc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# C99 keeps the fake backends / tests simple and portable.
target_compile_features(xcc PUBLIC c_std_99)

# dl* lives in libdl on glibc < 2.34 (e.g. CentOS 8) and in libc on newer
# glibc; CMAKE_DL_LIBS carries the right link library on each.
target_link_libraries(xcc PUBLIC ${CMAKE_DL_LIBS})

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(xcc PRIVATE -Wall -Wextra)
endif()

# --- install ----------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS xcc
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES docs/API.md docs/BACKENDS.md docs/SUPPORT_MATRIX.md
        DESTINATION ${CMAKE_INSTALL_DOCDIR}/xccl)

# --- examples ---------------------------------------------------------------
if(XCCL_BUILD_EXAMPLES)
    add_executable(minimal examples/minimal.c)
    target_link_libraries(minimal PRIVATE xcc::xcc)
endif()

# --- tests ------------------------------------------------------------------
if(XCCL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
