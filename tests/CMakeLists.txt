function(InitializeTest test_name file)
    add_executable(${test_name} ${file})
    set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY .)
    target_link_libraries(${test_name} digital_rf)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

InitializeTest(test_rf_write_hdf5 src/test_rf_write_hdf5.c)
