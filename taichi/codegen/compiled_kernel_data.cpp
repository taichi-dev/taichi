#include "compiled_kernel_data.h"

#include "picosha2.h"

namespace taichi::lang {

static CompiledKernelData::Err translate_err(CompiledKernelDataFile::Err err) {
  switch (err) {
    case CompiledKernelDataFile::Err::kNoError:
      return CompiledKernelData::Err::kNoError;
    case CompiledKernelDataFile::Err::kNotTicFile:
      return CompiledKernelData::Err::kNotTicFile;
    case CompiledKernelDataFile::Err::kCorruptedFile:
      return CompiledKernelData::Err::kCorruptedFile;
    case CompiledKernelDataFile::Err::kOutOfMemeory:
      return CompiledKernelData::Err::kOutOfMemeory;
    case CompiledKernelDataFile::Err::kIOStreamError:
      return CompiledKernelData::Err::kIOStreamError;
  }
  return CompiledKernelData::Err::kUnknown;
}

CompiledKernelDataFile::Err CompiledKernelDataFile::dump(std::ostream &os) {
  try {
    update_hash();
    std::uint32_t arch = static_cast<std::uint32_t>(arch_);
    std::uint64_t metadata_size = metadata_.size();
    std::uint64_t src_code_size = src_code_.size();
    bool io_success =
        os.write(head_, std::size(head_)) &&
        os.write((const char *)&arch_, sizeof(arch_)) &&
        os.write((const char *)&metadata_size, sizeof(metadata_size)) &&
        os.write((const char *)&src_code_size, sizeof(src_code_size)) &&
        os.write((const char *)metadata_.data(), metadata_size) &&
        os.write((const char *)src_code_.data(), src_code_size) &&
        os.write((const char *)hash_.data(), kHashSize);
    if (!io_success) {
      return Err::kIOStreamError;
    }
  } catch (std::bad_alloc &) {
    return Err::kOutOfMemeory;
  }
  return Err::kNoError;
}

CompiledKernelDataFile::Err CompiledKernelDataFile::load(std::istream &is) {
  try {
    if (!is.read(head_, std::size(head_))) {
      return Err::kIOStreamError;
    } else if (std::strncmp(head_, kHeadStr, kHeadSize) != 0) {
      return Err::kNotTicFile;
    }
    std::uint32_t arch;
    std::uint64_t metadata_size;
    std::uint64_t src_code_size;
    bool io_success = is.read((char *)&arch, sizeof(arch)) &&
                      is.read((char *)&metadata_size, sizeof(metadata_size)) &&
                      is.read((char *)&src_code_size, sizeof(src_code_size));
    if (!io_success) {
      return Err::kIOStreamError;
    }
    arch_ = static_cast<Arch>(arch);
    metadata_.resize(metadata_size);
    src_code_.resize(src_code_size);
    hash_.resize(kHashSize);
    io_success = is.read((char *)metadata_.data(), metadata_size) &&
                 is.read((char *)src_code_.data(), src_code_size) &&
                 is.read((char *)hash_.data(), kHashSize);
    if (!io_success) {
      return Err::kIOStreamError;
    }
    if (update_hash()) {
      return Err::kCorruptedFile;
    }
  } catch (std::bad_alloc &) {
    return Err::kOutOfMemeory;
  }
  return Err::kNoError;
}

bool CompiledKernelDataFile::update_hash() {
  picosha2::hash256_one_by_one hasher;
  hasher.process(metadata_.begin(), metadata_.end());
  hasher.process(src_code_.begin(), src_code_.end());
  hasher.finish();
  auto hash = picosha2::get_hash_hex_string(hasher);
  if (hash == hash_) {
    return false;
  }
  hash_ = std::move(hash);
  TI_ASSERT(hash_.size() == kHashSize);
  return true;
}

// FIXME: Uncomment after impl LLVMCompiledKernelData
// #if !defined(TI_WITH_LLVM)
CompiledKernelData::Creator *const CompiledKernelData::llvm_creator = nullptr;
// #endif

// FIXME: Uncomment after impl SpirvCompiledKernelData
// #if !defined(TI_WITH_VULKAN) && !defined(TI_WITH_OPENGL) && \
//     !defined(TI_WITH_DX11) && !defined(TI_WITH_METAL)
CompiledKernelData::Creator *const CompiledKernelData::spriv_creator = nullptr;
// #endif

CompiledKernelData::Err CompiledKernelData::load(std::istream &is) {
  try {
    Err err = Err::kNoError;
    CompiledKernelDataFile file;
    if (err = translate_err(file.load(is)); err != Err::kNoError) {
      return err;
    }
    return load_impl(file);
  } catch (std::bad_alloc &) {
    return Err::kOutOfMemeory;
  }
}

CompiledKernelData::Err CompiledKernelData::dump(std::ostream &os) const {
  try {
    Err err = Err::kNoError;
    CompiledKernelDataFile file;
    if (err = dump_impl(file); err != Err::kNoError) {
      return err;
    }
    return translate_err(file.dump(os));
  } catch (std::bad_alloc &) {
    return Err::kOutOfMemeory;
  }
}

// static functions
std::unique_ptr<CompiledKernelData> CompiledKernelData::load(std::istream &is,
                                                             Err *p_err) {
  Err err = Err::kNoError;
  CompiledKernelDataFile file;
  std::unique_ptr<CompiledKernelData> result{nullptr};
  try {
    err = translate_err(file.load(is));
    if (err == Err::kNoError) {
      result = create(file.arch(), err);
    }
    if (err == Err::kNoError) {
      TI_ASSERT(result);
      err = result->load_impl(file);
    }
  } catch (std::bad_alloc &) {
    err = Err::kOutOfMemeory;
  }
  if (p_err) {
    *p_err = err;
  }
  return result;
}

std::unique_ptr<CompiledKernelData> CompiledKernelData::create(Arch arch,
                                                               Err &err) {
  err = Err::kUnknown;
  if (arch_uses_llvm(arch)) {
    if (llvm_creator) {
      err = Err::kNoError;
      return llvm_creator();
    } else {
      err = Err::kTiWithoutLLVM;
    }
  } else if (arch_uses_spirv(arch)) {
    if (spriv_creator) {
      err = Err::kNoError;
      return spriv_creator();
    } else {
      err = Err::kTiWithoutSpirv;
    }
  }
  return nullptr;
}

}  // namespace taichi::lang
