// The code generator base class

#include "codegeb_base.h"
#if !defined(TI_PLATFORM_WINDOWS)
#include <xxhash.h>
#endif
#include <sstream>
#include <taichi/system/timer.h>

TLANG_NAMESPACE_BEGIN

std::string CodeGenBase::get_source_path(){TI_NOT_IMPLEMENTED}

std::string CodeGenBase::get_library_path() {
  TI_NOT_IMPLEMENTED
}

void CodeGenBase::write_source(){TI_NOT_IMPLEMENTED}

std::string CodeGenBase::get_source() {
  TI_NOT_IMPLEMENTED
}

void CodeGenBase::load_dll() {
  TI_NOT_IMPLEMENTED
}

void CodeGenBase::disassemble(){TI_NOT_IMPLEMENTED}

FunctionType CodeGenBase::load_function(){TI_NOT_IMPLEMENTED}

std::string CodeGenBase::get_source_name() {
  TI_NOT_IMPLEMENTED
}

void CodeGenBase::generate_binary(std::string extra_flags){TI_NOT_IMPLEMENTED}

CodeGenBase::~CodeGenBase() {
}

TLANG_NAMESPACE_END
