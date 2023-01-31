function(target_enable_function_level_linking TARGET)
    if(APPLE)
        target_link_options(${TARGET} PRIVATE -Wl,-dead_strip)
    elseif(MSVC) # WIN32
        target_link_options(${TARGET} PRIVATE /Gy)
    else() # Linux / *nix / gcc compatible
        target_link_options(${TARGET} PRIVATE -Wl,--gc-sections)
    endif()
endfunction()
