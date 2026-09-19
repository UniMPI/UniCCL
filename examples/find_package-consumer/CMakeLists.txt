# find_package-consumer - minimal downstream that links UniCCL via
# find_package(UniCCL CONFIG) and the exported unicc::unicc target.
#
# It is NOT part of the UniCCL superbuild (the top-level CMakeLists installs
# UniCCL; this is a standalone consumer to smoke-test the installed package).
# Build against an installed prefix:
#
#   cmake -S . -B build-consumer \
#       -DCMAKE_PREFIX_PATH=<install-prefix> \
#       -DUNICC_CONSUMER_BACKEND=<path-or-name>   # optional, see main.c
#   cmake --build build-consumer
#   UNICC_LIBRARY=... ./build-consumer/consumer   # smoke: init + version
#
# Everything a consumer needs arrives through the export: the install-layout
# include dir, c_std_99, the dl* link library and PIC are all attached to
# unicc::unicc, so no flag is set here.

cmake_minimum_required(VERSION 3.10)
project(unicc_consumer LANGUAGES C)

find_package(UniCCL CONFIG REQUIRED)

add_executable(consumer main.c)
target_link_libraries(consumer PRIVATE unicc::unicc)
