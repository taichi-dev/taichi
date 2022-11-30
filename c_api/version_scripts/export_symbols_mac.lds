# Note: this version script for MacOS should match the one for Linux (export_symbols_linux.ld)
_ti_* # C-API symbols starts with _ti_* on MacOS (conventionally, extern "C" symbols have a leading underscore)
__ZN4capi5utils* # matches "capi::utils::*" for C-API tests
