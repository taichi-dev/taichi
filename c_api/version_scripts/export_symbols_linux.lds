LIB_C_API_1.0 {
  global: # specify symbols to be export from libtaichi_c_api
    ti_*; # CAPI interface starts with "ti_*"
    _ZN4capi5utils*; # matches "capi::utils::*" for CAPI tests

  local:
    *; # all other symbols will be convert to local bind type whatsoever
};
