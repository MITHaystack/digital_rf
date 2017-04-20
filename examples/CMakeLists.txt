add_custom_target(examples)

function(BuildExample example_name file)
    add_executable(${example_name} ${file})
    add_dependencies(examples ${example_name})
    set_target_properties(${example_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY .)
    target_link_libraries(${example_name} digital_rf)
endfunction()

BuildExample(benchmark_rf_write_hdf5 src/benchmark_rf_write_hdf5.c)
BuildExample(example_rf_write_hdf5 src/example_rf_write_hdf5.c)
BuildExample(example_with_sigint_handler src/example_with_sigint_handler.c)
