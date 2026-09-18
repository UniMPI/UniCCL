# UniCCL unit test suite. Runs entirely on the host: the fixtures under
# fake_backends/ are fake .so libraries UniCCL dlopens as stand-ins for real
# NCCL/RCCL, so no GPU, CUDA or ROCm is required.

set(UNICC_TEST_FIXTURES_DIR ${CMAKE_CURRENT_BINARY_DIR}/fake)

# Build a fake backend fixture (a shared library exporting an NCCL/RCCL-shaped
# symbol set), optionally with compile-time variants.
function(add_fake_unicc_fixture target source)
    add_library(${target} SHARED ${source})
    target_compile_features(${target} PRIVATE c_std_99)
    set_target_properties(${target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${UNICC_TEST_FIXTURES_DIR}
        PREFIX "")
    foreach(def IN LISTS ARGN)
        target_compile_definitions(${target} PRIVATE ${def})
    endforeach()
endfunction()

add_fake_unicc_fixture(fake_nccl_identity                fake_backends/fake_nccl_identity.c)
add_fake_unicc_fixture(fake_nccl_missing_group_end       fake_backends/fake_nccl_identity.c UNICC_FAKE_OMIT_GROUP_END)
add_fake_unicc_fixture(fake_rccl_identity                fake_backends/fake_rccl_identity.c)

# --- test_loader ------------------------------------------------------------
add_executable(test_loader test_loader.c)
target_link_libraries(test_loader PRIVATE unicc::unicc)
add_dependencies(test_loader fake_nccl_identity fake_rccl_identity fake_nccl_missing_group_end)
add_test(NAME test_loader COMMAND test_loader
    --fake-nccl $<TARGET_FILE:fake_nccl_identity>
    --fake-rccl $<TARGET_FILE:fake_rccl_identity>
    --fake-nccl-missing $<TARGET_FILE:fake_nccl_missing_group_end>)
set_tests_properties(test_loader PROPERTIES LABELS "unit;fake;loader")

# --- test_vtable ------------------------------------------------------------
add_executable(test_vtable test_vtable.c)
target_link_libraries(test_vtable PRIVATE unicc::unicc)
add_dependencies(test_vtable fake_nccl_identity fake_nccl_missing_group_end)
add_test(NAME test_vtable COMMAND test_vtable
    --fake-nccl $<TARGET_FILE:fake_nccl_identity>
    --fake-nccl-missing $<TARGET_FILE:fake_nccl_missing_group_end>)
set_tests_properties(test_vtable PROPERTIES LABELS "unit;fake;vtable")

# --- test_api ---------------------------------------------------------------
add_executable(test_api test_api.c)
target_link_libraries(test_api PRIVATE unicc::unicc)
add_dependencies(test_api fake_nccl_identity fake_rccl_identity fake_nccl_missing_group_end)
add_test(NAME test_api COMMAND test_api
    --fake-nccl $<TARGET_FILE:fake_nccl_identity>
    --fake-rccl $<TARGET_FILE:fake_rccl_identity>
    --fake-nccl-missing $<TARGET_FILE:fake_nccl_missing_group_end>)
set_tests_properties(test_api PROPERTIES LABELS "unit;fake;api")

# --- integration (real backends; opt-in, driven by run_integration.sh) ------
if(UNICC_BUILD_TESTS_INTEGRATION)
    add_executable(test_integration
        integration/test_integration.c
        integration/devmem.c)
    target_include_directories(test_integration PRIVATE integration)
    target_link_libraries(test_integration PRIVATE unicc::unicc)
    set_target_properties(test_integration PROPERTIES LABELS "integration")
endif()
