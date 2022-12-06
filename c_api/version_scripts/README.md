Taichi C-API uses version script to control symbol exports from the shared library (https://github.com/taichi-dev/taichi/issues/6722)

Note that the two version scripts export_symbols_linux.ld and export_symbols_mac.ld should have consistent contents otherwise will cause symbols mismatch on different platforms.

CI will check for symbol leakage automatically, the regular expression rule of which should also stay consistent with the version scripts. CI scripts can be found at "check-c-api-export-symbols" function in .github/workflows/scripts/aot-demo.sh
