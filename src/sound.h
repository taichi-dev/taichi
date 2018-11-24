#pragma once
#include <taichi/util.h>
#include <cmath>
#include <fstream>
#include <iostream>

TC_NAMESPACE_BEGIN

// http://soundfile.sapp.org/doc/WaveFormat/

using namespace std;
namespace little_endian_io {
template <typename Word>
std::ostream &write_word(std::ostream &outs,
                         Word value,
                         unsigned size = sizeof(Word)) {
  for (; size; --size, value >>= 8)
    outs.put(static_cast<char>(value & 0xFF));
  return outs;
}
}
using namespace little_endian_io;

class WaveFile {
 public:
  int sample_rate = 44100;
  static constexpr real max_amplitude = 32760_f;  // "volume"
  std::string file_name;
  std::vector<real> data;

  real current_t = 0;

  TC_IO_DECL {
    TC_IO(sample_rate);
    TC_IO(file_name);
    TC_IO(data);
    TC_IO(current_t);
  };

  void add_sound(real dt, real magnitude) {
    int begin = current_t * sample_rate;
    int end = (current_t + dt) * sample_rate;
    for (int i = begin; i < end; i++) {
      data.push_back(rand() * magnitude);
    }
    current_t += dt;
  }

  static void write_header(ofstream &f) {
    // Write the file headers
    f << "RIFF----WAVEfmt ";   // (chunk size to be filled in later)
    write_word(f, 16, 4);      // no extension data
    write_word(f, 1, 2);       // PCM - integer samples
    write_word(f, 2, 2);       // two channels (stereo file)
    write_word(f, 44100, 4);   // samples per second (Hz)
    write_word(f, 176400, 4);  // (Sample Rate * BitsPerSample * Channels) / 8
    write_word(f, 4, 2);  // data block size (size of two integer samples, one
    // for each channel, in bytes)
    write_word(f, 16, 2);  // number of bits per sample (use a multiple of 8)
  }

  static void write_data(ofstream &f, const std::vector<real> &data) {
    // Write the data chunk header
    size_t data_chunk_pos = f.tellp();
    f << "data----";  // (chunk size to be filled in later)

    for (int i = 0; i < (int)data.size(); i++) {
      real value = max_amplitude * clamp(data[i], 0.0_f, 1.0_f);
      write_word(f, (int)value, 2);
      write_word(f, (int)value, 2);
    }

    // (We'll need the final file size to fix the chunk sizes above)
    size_t file_length = f.tellp();

    // Fix the data chunk header to contain the data size
    f.seekp(data_chunk_pos + 4);
    write_word(f, file_length - data_chunk_pos + 8);

    // Fix the file header to contain the proper RIFF chunk size, which is (file
    // size - 8) bytes
    f.seekp(0 + 4);
    write_word(f, file_length - 8, 4);
  }

  WaveFile() {
  }

  WaveFile(const std::string &file_name) : file_name(file_name) {
    TC_ASSERT(ends_with(file_name, ".wav"));
  }

  void initialize(const std::string &file_name) {
    this->file_name = file_name;
  }

  void flush() const {
    write(file_name, data);
  }

  static void write(const std::string &file_name,
                    const std::vector<real> &data) {
    auto f = ofstream(file_name, ios::binary);
    write_header(f);
    write_data(f, data);
  }
};

TC_NAMESPACE_END
