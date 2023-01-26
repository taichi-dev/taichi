function(target_enable_function_level_linking TARGET)
    if(APPLE)
        target_link_options(${TARGET} PRIVATE -Wl,-dead_strip)
    elseif(MSVC) # LINUX AND WIN32
        target_link_options(${TARGET} PRIVATE /Gy)
    else()
        target_link_options(${TARGET} PRIVATE -Wl,--gc-sections)
    endif()
endfunction()
