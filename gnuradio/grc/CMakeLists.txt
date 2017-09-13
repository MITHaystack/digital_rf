find_program(PYTHON "python" REQUIRED)

macro(GEN_BLOCK_XML _generator _xml_block)
    set(generator ${CMAKE_CURRENT_SOURCE_DIR}/${_generator})
    set(xml_block ${CMAKE_CURRENT_BINARY_DIR}/${_xml_block})
    list(APPEND generated_blocks ${xml_block})
    add_custom_command(
        DEPENDS ${generator} OUTPUT ${xml_block}
        COMMAND ${PYTHON} ${generator} ${xml_block}
    )
endmacro(GEN_BLOCK_XML)


GEN_BLOCK_XML(gen_gr_drf_digital_rf_sink.py gr_drf_digital_rf_sink.xml)
GEN_BLOCK_XML(gen_gr_drf_digital_rf_source.py gr_drf_digital_rf_source.xml)

add_custom_target(gr_drf_grc_xml_blocks ALL DEPENDS ${generated_blocks})
install(FILES
    ${generated_blocks}
    drf_digital_rf_sink.xml
    gr_drf_digital_rf_channel_sink.xml
    gr_drf_digital_rf_channel_source.xml
    DESTINATION ${GRC_BLOCKS_DIR}
)
