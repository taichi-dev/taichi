#include "taichi/common/zip.h"
#include "taichi/common/miniz.h"

namespace taichi {
namespace zip {

bool ZipArchive::try_from_bytes(const void *data, size_t size, ZipArchive &ar) {
  mz_bool succ = MZ_TRUE;
  ar.file_dict.clear();

  mz_zip_archive zip;
  mz_zip_zero_struct(&zip);

  {
    succ &= mz_zip_reader_init_mem(&zip, data, size, 0);
    if (succ != MZ_TRUE) {
      goto fail;
    }
  }

  {
    mz_uint nfile = mz_zip_reader_get_num_files(&zip);
    for (mz_uint i = 0; i < nfile; ++i) {
      mz_zip_archive_file_stat file_stat;
      succ &= mz_zip_reader_file_stat(&zip, i, &file_stat);
      if (succ != MZ_TRUE) {
        goto fail;
      }

      std::vector<uint8_t> file_data(file_stat.m_uncomp_size);
      succ &= mz_zip_reader_extract_to_mem(&zip, i, file_data.data(),
                                           file_data.size(), 0);
      if (succ != MZ_TRUE) {
        goto fail;
      }
      ar.file_dict[file_stat.m_filename] = std::move(file_data);
    }
  }

fail:
  succ &= mz_zip_reader_end(&zip);

  return succ;
}

}  // namespace zip
}  // namespace taichi
