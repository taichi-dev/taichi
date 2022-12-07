function(configure_target_linker TARGET_NAME USE_MOLD)

if (${USE_MOLD})
    target_link_options(${TARGET_NAME} PRIVATE -fuse-ld=mold)
elseif(WIN32 OR LINUX)
    # Use "lld" as default linker instead of "ld" because:
    # 1. lld is a drop-in replacement for ld
    # 2. lld is way more performant than ld
    # 3. lld is simpler to use, whereas ld has strict restrictions on link order
    #    and you will have to play with --start-group/--end-group to work around it
    #
    # Ref1: https://stackoverflow.com/questions/29361801/is-the-lld-linker-a-drop-in-replacement-for-ld-and-gold
    # Ref2: https://community.haxe.org/t/hxcpp-on-linux-tip-replace-ld-linker-with-lld-to-seriously-improve-compilation-time/2134
    target_link_options(${TARGET_NAME} PRIVATE -fuse-ld=lld)
    if(WIN32 AND CMAKE_BUILD_TYPE EQUAL "RelWithDebInfo")
        # -debug for LLD generates PDB files
        target_link_options(${TARGET_NAME} PRIVATE -Wl,-debug)
    endif()
endif() # USE_MOLD

endfunction()
