// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Jacob <benoitjacob@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <cmath>
#include <cassert>
#include <cstring>
#include <memory>

#include <Eigen/Core>

using namespace std;

const int default_precision = 4;

// see --only-cubic-sizes
bool only_cubic_sizes = false;

// see --dump-tables
bool dump_tables = false;

uint8_t log2_pot(size_t x) {
  size_t l = 0;
  while (x >>= 1) l++;
  return l;
}

uint16_t compact_size_triple(size_t k, size_t m, size_t n)
{
  return (log2_pot(k) << 8) | (log2_pot(m) << 4) | log2_pot(n);
}

// just a helper to store a triple of K,M,N sizes for matrix product
struct size_triple_t
{
  uint16_t k, m, n;
  size_triple_t() : k(0), m(0), n(0) {}
  size_triple_t(size_t _k, size_t _m, size_t _n) : k(_k), m(_m), n(_n) {}
  size_triple_t(const size_triple_t& o) : k(o.k), m(o.m), n(o.n) {}
  size_triple_t(uint16_t compact)
  {
    k = 1 << ((compact & 0xf00) >> 8);
    m = 1 << ((compact & 0x0f0) >> 4);
    n = 1 << ((compact & 0x00f) >> 0);
  }
  bool is_cubic() const { return k == m && m == n; }
};

ostream& operator<<(ostream& s, const size_triple_t& t)
{
  return s << "(" << t.k << ", " << t.m << ", " << t.n << ")";
}

struct inputfile_entry_t
{
  uint16_t product_size;
  uint16_t pot_block_size;
  size_triple_t nonpot_block_size;
  float gflops;
};

struct inputfile_t
{
  enum class type_t {
    unknown,
    all_pot_sizes,
    default_sizes
  };

  string filename;
  vector<inputfile_entry_t> entries;
  type_t type;

  inputfile_t(const string& fname)
    : filename(fname)
    , type(type_t::unknown)
  {
    ifstream stream(filename);
    if (!stream.is_open()) {
      cerr << "couldn't open input file: " << filename << endl;
      exit(1);
    }
    string line;
    while (getline(stream, line)) {
      if (line.empty()) continue;
      if (line.find("BEGIN MEASUREMENTS ALL POT SIZES") == 0) {
        if (type != type_t::unknown) {
          cerr << "Input file " << filename << " contains redundant BEGIN MEASUREMENTS lines";
          exit(1);
        }
        type = type_t::all_pot_sizes;
        continue;
      }
      if (line.find("BEGIN MEASUREMENTS DEFAULT SIZES") == 0) {
        if (type != type_t::unknown) {
          cerr << "Input file " << filename << " contains redundant BEGIN MEASUREMENTS lines";
          exit(1);
        }
        type = type_t::default_sizes;
        continue;
      }
      

      if (type == type_t::unknown) {
        continue;
      }
      switch(type) {
        case type_t::all_pot_sizes: {
          unsigned int product_size, block_size;
          float gflops;
          int sscanf_result =
            sscanf(line.c_str(), "%x %x %f",
                   &product_size,
                   &block_size,
                   &gflops);
          if (3 != sscanf_result ||
              !product_size ||
              product_size > 0xfff ||
              !block_size ||
              block_size > 0xfff ||
              !isfinite(gflops))
          {
            cerr << "ill-formed input file: " << filename << endl;
            cerr << "offending line:" << endl << line << endl;
            exit(1);
          }
          if (only_cubic_sizes && !size_triple_t(product_size).is_cubic()) {
            continue;
          }
          inputfile_entry_t entry;
          entry.product_size = uint16_t(product_size);
          entry.pot_block_size = uint16_t(block_size);
          entry.gflops = gflops;
          entries.push_back(entry);
          break;
        }
        case type_t::default_sizes: {
          unsigned int product_size;
          float gflops;
          int bk, bm, bn;
          int sscanf_result =
            sscanf(line.c_str(), "%x default(%d, %d, %d) %f",
                   &product_size,
                   &bk, &bm, &bn,
                   &gflops);
          if (5 != sscanf_result ||
              !product_size ||
              product_size > 0xfff ||
              !isfinite(gflops))
          {
            cerr << "ill-formed input file: " << filename << endl;
            cerr << "offending line:" << endl << line << endl;
            exit(1);
          }
          if (only_cubic_sizes && !size_triple_t(product_size).is_cubic()) {
            continue;
          }
          inputfile_entry_t entry;
          entry.product_size = uint16_t(product_size);
          entry.pot_block_size = 0;
          entry.nonpot_block_size = size_triple_t(bk, bm, bn);
          entry.gflops = gflops;
          entries.push_back(entry);
          break;
        }
        
        default:
          break;
      }
    }
    stream.close();
    if (type == type_t::unknown) {
      cerr << "Unrecognized input file " << filename << endl;
      exit(1);
    }
    if (entries.empty()) {
      cerr << "didn't find any measurements in input file: " << filename << endl;
      exit(1);
    }
  }
};

struct preprocessed_inputfile_entry_t
{
  uint16_t product_size;
  uint16_t block_size;

  float efficiency;
};

bool lower_efficiency(const preprocessed_inputfile_entry_t& e1, const preprocessed_inputfile_entry_t& e2)
{
  return e1.efficiency < e2.efficiency;
}

struct preprocessed_inputfile_t
{
  string filename;
  vector<preprocessed_inputfile_entry_t> entries;

  preprocessed_inputfile_t(const inputfile_t& inputfile)
    : filename(inputfile.filename)
  {
    if (inputfile.type != inputfile_t::type_t::all_pot_sizes) {
      abort();
    }
    auto it = inputfile.entries.begin();
    auto it_first_with_given_product_size = it;
    while (it != inputfile.entries.end()) {
      ++it;
      if (it == inputfile.entries.end() ||
        it->product_size != it_first_with_given_product_size->product_size)
      {
        import_input_file_range_one_product_size(it_first_with_given_product_size, it);
        it_first_with_given_product_size = it;
      }
    }
  }

private:
  void import_input_file_range_one_product_size(
    const vector<inputfile_entry_t>::const_iterator& begin,
    const vector<inputfile_entry_t>::const_iterator& end)
  {
    uint16_t product_size = begin->product_size;
    float max_gflops = 0.0f;
    for (auto it = begin; it != end; ++it) {
      if (it->product_size != product_size) {
        cerr << "Unexpected ordering of entries in " << filename << endl;
        cerr << "(Expected all entries for product size " << hex << product_size << dec << " to be grouped)" << endl;
        exit(1);
      }
      max_gflops = max(max_gflops, it->gflops);
    }
    for (auto it = begin; it != end; ++it) {
      preprocessed_inputfile_entry_t entry;
      entry.product_size = it->product_size;
      entry.block_size = it->pot_block_size;
      entry.efficiency = it->gflops / max_gflops;
      entries.push_back(entry);
    }
  }
};

void check_all_files_in_same_exact_order(
       const vector<preprocessed_inputfile_t>& preprocessed_inputfiles)
{
  if (preprocessed_inputfiles.empty()) {
    return;
  }

  const preprocessed_inputfile_t& first_file = preprocessed_inputfiles[0];
  const size_t num_entries = first_file.entries.size();

  for (size_t i = 0; i < preprocessed_inputfiles.size(); i++) {
    if (preprocessed_inputfiles[i].entries.size() != num_entries) {
      cerr << "these files have different number of entries: "
           << preprocessed_inputfiles[i].filename
           << " and "
           << first_file.filename
           << endl;
      exit(1);
    }
  }

  for (size_t entry_index = 0; entry_index < num_entries; entry_index++) {
    const uint16_t entry_product_size = first_file.entries[entry_index].product_size;
    const uint16_t entry_block_size = first_file.entries[entry_index].block_size;
    for (size_t file_index = 0; file_index < preprocessed_inputfiles.size(); file_index++) {
      const preprocessed_inputfile_t& cur_file = preprocessed_inputfiles[file_index];
      if (cur_file.entries[entry_index].product_size != entry_product_size ||
          cur_file.entries[entry_index].block_size != entry_block_size)
      {
        cerr << "entries not in same order between these files: "
             << first_file.filename
             << " and "
             << cur_file.filename
             << endl;
        exit(1);
      }
    }
  }
}

float efficiency_of_subset(
        const vector<preprocessed_inputfile_t>& preprocessed_inputfiles,
        const vector<size_t>& subset)
{
  if (subset.size() <= 1) {
    return 1.0f;
  }
  const preprocessed_inputfile_t& first_file = preprocessed_inputfiles[subset[0]];
  const size_t num_entries = first_file.entries.size();
  float efficiency = 1.0f;
  size_t entry_index = 0;
  size_t first_entry_index_with_this_product_size = 0;
  uint16_t product_size = first_file.entries[0].product_size;
  while (entry_index < num_entries) {
    ++entry_index;
    if (entry_index == num_entries ||
        first_file.entries[entry_index].product_size != product_size)
    {
      float efficiency_this_product_size = 0.0f;
      for (size_t e = first_entry_index_with_this_product_size; e < entry_index; e++) {
        float efficiency_this_entry = 1.0f;
        for (auto i = subset.begin(); i != subset.end(); ++i) {
          efficiency_this_entry = min(efficiency_this_entry, preprocessed_inputfiles[*i].entries[e].efficiency);
        }
        efficiency_this_product_size = max(efficiency_this_product_size, efficiency_this_entry);
      }
      efficiency = min(efficiency, efficiency_this_product_size);
      if (entry_index < num_entries) {
        first_entry_index_with_this_product_size = entry_index;
        product_size = first_file.entries[entry_index].product_size;
      }
    }
  }

  return efficiency;
}

void dump_table_for_subset(
        const vector<preprocessed_inputfile_t>& preprocessed_inputfiles,
        const vector<size_t>& subset)
{
  const preprocessed_inputfile_t& first_file = preprocessed_inputfiles[subset[0]];
  const size_t num_entries = first_file.entries.size();
  size_t entry_index = 0;
  size_t first_entry_index_with_this_product_size = 0;
  uint16_t product_size = first_file.entries[0].product_size;
  size_t i = 0;
  size_triple_t min_product_size(first_file.entries.front().product_size);
  size_triple_t max_product_size(first_file.entries.back().product_size);
  if (!min_product_size.is_cubic() || !max_product_size.is_cubic()) {
    abort();
  }
  if (only_cubic_sizes) {
    cerr << "Can't generate tables with --only-cubic-sizes." << endl;
    abort();
  }
  cout << "struct LookupTable {" << endl;
  cout << "  static const size_t BaseSize = " << min_product_size.k << ";" << endl;
  const size_t NumSizes = log2_pot(max_product_size.k / min_product_size.k) + 1;
  const size_t TableSize = NumSizes * NumSizes * NumSizes;
  cout << "  static const size_t NumSizes = " << NumSizes << ";" << endl;
  cout << "  static const unsigned short* Data() {" << endl;
  cout << "    static const unsigned short data[" << TableSize << "] = {";
  while (entry_index < num_entries) {
    ++entry_index;
    if (entry_index == num_entries ||
        first_file.entries[entry_index].product_size != product_size)
    {
      float best_efficiency_this_product_size = 0.0f;
      uint16_t best_block_size_this_product_size = 0;
      for (size_t e = first_entry_index_with_this_product_size; e < entry_index; e++) {
        float efficiency_this_entry = 1.0f;
        for (auto i = subset.begin(); i != subset.end(); ++i) {
          efficiency_this_entry = min(efficiency_this_entry, preprocessed_inputfiles[*i].entries[e].efficiency);
        }
        if (efficiency_this_entry > best_efficiency_this_product_size) {
          best_efficiency_this_product_size = efficiency_this_entry;
          best_block_size_this_product_size = first_file.entries[e].block_size;
        }
      }
      if ((i++) % NumSizes) {
        cout << " ";
      } else {
        cout << endl << "      ";
      }
      cout << "0x" << hex << best_block_size_this_product_size << dec;
      if (entry_index < num_entries) {
        cout << ",";
        first_entry_index_with_this_product_size = entry_index;
        product_size = first_file.entries[entry_index].product_size;
      }
    }
  }
  if (i != TableSize) {
    cerr << endl << "Wrote " << i << " table entries, expected " << TableSize << endl;
    abort();
  }
  cout << endl << "    };" << endl;
  cout << "    return data;" << endl;
  cout << "  }" << endl;
  cout << "};" << endl;
}

float efficiency_of_partition(
        const vector<preprocessed_inputfile_t>& preprocessed_inputfiles,
        const vector<vector<size_t>>& partition)
{
  float efficiency = 1.0f;
  for (auto s = partition.begin(); s != partition.end(); ++s) {
    efficiency = min(efficiency, efficiency_of_subset(preprocessed_inputfiles, *s));
  }
  return efficiency;
}

void make_first_subset(size_t subset_size, vector<size_t>& out_subset, size_t set_size)
{
  assert(subset_size >= 1 && subset_size <= set_size);
  out_subset.resize(subset_size);
  for (size_t i = 0; i < subset_size; i++) {
    out_subset[i] = i;
  }
}

bool is_last_subset(const vector<size_t>& subset, size_t set_size)
{
  return subset[0] == set_size - subset.size();
}

void next_subset(vector<size_t>& inout_subset, size_t set_size)
{
  if (is_last_subset(inout_subset, set_size)) {
    cerr << "iterating past the last subset" << endl;
    abort();
  }
  size_t i = 1;
  while (inout_subset[inout_subset.size() - i] == set_size - i) {
    i++;
    assert(i <= inout_subset.size());
  }
  size_t first_index_to_change = inout_subset.size() - i;
  inout_subset[first_index_to_change]++;
  size_t p = inout_subset[first_index_to_change];
  for (size_t j = first_index_to_change + 1; j < inout_subset.size(); j++) {
    inout_subset[j] = ++p;
  }
}

const size_t number_of_subsets_limit = 100;
const size_t always_search_subsets_of_size_at_least = 2;

bool is_number_of_subsets_feasible(size_t n, size_t p)
{ 
  assert(n>0 && p>0 && p<=n);
  uint64_t numerator = 1, denominator = 1;
  for (size_t i = 0; i < p; i++) {
    numerator *= n - i;
    denominator *= i + 1;
    if (numerator > denominator * number_of_subsets_limit) {
      return false;
    }
  }
  return true;
}

size_t max_feasible_subset_size(size_t n)
{
  assert(n > 0);
  const size_t minresult = min<size_t>(n-1, always_search_subsets_of_size_at_least);
  for (size_t p = 1; p <= n - 1; p++) {
    if (!is_number_of_subsets_feasible(n, p+1)) {
      return max(p, minresult);
    }
  }
  return n - 1;
}

void find_subset_with_efficiency_higher_than(
       const vector<preprocessed_inputfile_t>& preprocessed_inputfiles,
       float required_efficiency_to_beat,
       vector<size_t>& inout_remainder,
       vector<size_t>& out_subset)
{
  out_subset.resize(0);

  if (required_efficiency_to_beat >= 1.0f) {
    cerr << "can't beat efficiency 1." << endl;
    abort();
  }

  while (!inout_remainder.empty()) {

    vector<size_t> candidate_indices(inout_remainder.size());
    for (size_t i = 0; i < candidate_indices.size(); i++) {
      candidate_indices[i] = i;
    }

    size_t candidate_indices_subset_size = max_feasible_subset_size(candidate_indices.size());
    while (candidate_indices_subset_size >= 1) {
      vector<size_t> candidate_indices_subset;
      make_first_subset(candidate_indices_subset_size,
                        candidate_indices_subset,
                        candidate_indices.size());

      vector<size_t> best_candidate_indices_subset;
      float best_efficiency = 0.0f;
      vector<size_t> trial_subset = out_subset;
      trial_subset.resize(out_subset.size() + candidate_indices_subset_size);
      while (true)
      {
        for (size_t i = 0; i < candidate_indices_subset_size; i++) {
          trial_subset[out_subset.size() + i] = inout_remainder[candidate_indices_subset[i]];
        }
        
        float trial_efficiency = efficiency_of_subset(preprocessed_inputfiles, trial_subset);
        if (trial_efficiency > best_efficiency) {
          best_efficiency = trial_efficiency;
          best_candidate_indices_subset = candidate_indices_subset;
        }
        if (is_last_subset(candidate_indices_subset, candidate_indices.size())) {
          break;
        }
        next_subset(candidate_indices_subset, candidate_indices.size());
      }
       
      if (best_efficiency > required_efficiency_to_beat) {
        for (size_t i = 0; i < best_candidate_indices_subset.size(); i++) {
          candidate_indices[i] = candidate_indices[best_candidate_indices_subset[i]];
        }
        candidate_indices.resize(best_candidate_indices_subset.size());
      }
      candidate_indices_subset_size--;
    }
      
    size_t candidate_index = candidate_indices[0];
    auto candidate_iterator = inout_remainder.begin() + candidate_index;
    vector<size_t> trial_subset = out_subset;

    trial_subset.push_back(*candidate_iterator);
    float trial_efficiency = efficiency_of_subset(preprocessed_inputfiles, trial_subset);
    if (trial_efficiency > required_efficiency_to_beat) {
      out_subset.push_back(*candidate_iterator);
      inout_remainder.erase(candidate_iterator);
    } else {
      break;
    }
  }
}

void find_partition_with_efficiency_higher_than(
       const vector<preprocessed_inputfile_t>& preprocessed_inputfiles,
       float required_efficiency_to_beat,
       vector<vector<size_t>>& out_partition)
{
  out_partition.resize(0);

  vector<size_t> remainder;
  for (size_t i = 0; i < preprocessed_inputfiles.size(); i++) {
    remainder.push_back(i);
  }

  while (!remainder.empty()) {
    vector<size_t> new_subset;
    find_subset_with_efficiency_higher_than(
      preprocessed_inputfiles,
      required_efficiency_to_beat,
      remainder,
      new_subset);
    out_partition.push_back(new_subset);
  }
}

void print_partition(
       const vector<preprocessed_inputfile_t>& preprocessed_inputfiles,
       const vector<vector<size_t>>& partition)
{
  float efficiency = efficiency_of_partition(preprocessed_inputfiles, partition);
  cout << "Partition into " << partition.size() << " subsets for " << efficiency * 100.0f << "% efficiency"  << endl;
  for (auto subset = partition.begin(); subset != partition.end(); ++subset) {
    cout << "  Subset " << (subset - partition.begin())
         << ", efficiency " << efficiency_of_subset(preprocessed_inputfiles, *subset) * 100.0f << "%:"
         << endl;
    for (auto file = subset->begin(); file != subset->end(); ++file) {
      cout << "    " << preprocessed_inputfiles[*file].filename << endl;
    }
    if (dump_tables) {
      cout << "  Table:" << endl;
      dump_table_for_subset(preprocessed_inputfiles, *subset);
    }
  }
  cout << endl;
}

struct action_t
{
  virtual const char* invokation_name() const { abort(); return nullptr; }
  virtual void run(const vector<string>&) const { abort(); }
  virtual ~action_t() {}
};

struct partition_action_t : action_t
{
  virtual const char* invokation_name() const override { return "partition"; }
  virtual void run(const vector<string>& input_filenames) const override
  {
    vector<preprocessed_inputfile_t> preprocessed_inputfiles;

    if (input_filenames.empty()) {
      cerr << "The " << invokation_name() << " action needs a list of input files." << endl;
      exit(1);
    }

    for (auto it = input_filenames.begin(); it != input_filenames.end(); ++it) {
      inputfile_t inputfile(*it);
      switch (inputfile.type) {
        case inputfile_t::type_t::all_pot_sizes:
          preprocessed_inputfiles.emplace_back(inputfile);
          break;
        case inputfile_t::type_t::default_sizes:
          cerr << "The " << invokation_name() << " action only uses measurements for all pot sizes, and "
               << "has no use for " << *it << " which contains measurements for default sizes." << endl;
          exit(1);
          break;
        default:
          cerr << "Unrecognized input file: " << *it << endl;
          exit(1);
      }
    }

    check_all_files_in_same_exact_order(preprocessed_inputfiles);

    float required_efficiency_to_beat = 0.0f;
    vector<vector<vector<size_t>>> partitions;
    cerr << "searching for partitions...\r" << flush;
    while (true)
    {
      vector<vector<size_t>> partition;
      find_partition_with_efficiency_higher_than(
        preprocessed_inputfiles,
        required_efficiency_to_beat,
        partition);
      float actual_efficiency = efficiency_of_partition(preprocessed_inputfiles, partition);
      cerr << "partition " << preprocessed_inputfiles.size() << " files into " << partition.size()
           << " subsets for " << 100.0f * actual_efficiency
           << " % efficiency"
           << "                  \r" << flush;
      partitions.push_back(partition);
      if (partition.size() == preprocessed_inputfiles.size() || actual_efficiency == 1.0f) {
        break;
      }
      required_efficiency_to_beat = actual_efficiency;
    }
    cerr << "                                                                  " << endl;
    while (true) {
      bool repeat = false;
      for (size_t i = 0; i < partitions.size() - 1; i++) {
        if (partitions[i].size() >= partitions[i+1].size()) {
          partitions.erase(partitions.begin() + i);
          repeat = true;
          break;
        }
      }
      if (!repeat) {
        break;
      }
    }
    for (auto it = partitions.begin(); it != partitions.end(); ++it) {
      print_partition(preprocessed_inputfiles, *it);
    }
  }
};

struct evaluate_defaults_action_t : action_t
{
  struct results_entry_t {
    uint16_t product_size;
    size_triple_t default_block_size;
    uint16_t best_pot_block_size;
    float default_gflops;
    float best_pot_gflops;
    float default_efficiency;
  };
  friend ostream& operator<<(ostream& s, const results_entry_t& entry)
  {
    return s
      << "Product size " << size_triple_t(entry.product_size)
      << ": default block size " << entry.default_block_size
      << " -> " << entry.default_gflops
      << " GFlop/s = " << entry.default_efficiency * 100.0f << " %"
      << " of best POT block size " << size_triple_t(entry.best_pot_block_size)
      << " -> " << entry.best_pot_gflops
      << " GFlop/s" << dec;
  }
  static bool lower_efficiency(const results_entry_t& e1, const results_entry_t& e2) {
    return e1.default_efficiency < e2.default_efficiency;
  }
  virtual const char* invokation_name() const override { return "evaluate-defaults"; }
  void show_usage_and_exit() const
  {
    cerr << "usage: " << invokation_name() << " default-sizes-data all-pot-sizes-data" << endl;
    cerr << "checks how well the performance with default sizes compares to the best "
         << "performance measured over all POT sizes." << endl;
    exit(1);
  }
  virtual void run(const vector<string>& input_filenames) const override
  {
    if (input_filenames.size() != 2) {
      show_usage_and_exit();
    }
    inputfile_t inputfile_default_sizes(input_filenames[0]);
    inputfile_t inputfile_all_pot_sizes(input_filenames[1]);
    if (inputfile_default_sizes.type != inputfile_t::type_t::default_sizes) {
      cerr << inputfile_default_sizes.filename << " is not an input file with default sizes." << endl;
      show_usage_and_exit();
    }
    if (inputfile_all_pot_sizes.type != inputfile_t::type_t::all_pot_sizes) {
      cerr << inputfile_all_pot_sizes.filename << " is not an input file with all POT sizes." << endl;
      show_usage_and_exit();
    }
    vector<results_entry_t> results;
    vector<results_entry_t> cubic_results;
    
    uint16_t product_size = 0;
    auto it_all_pot_sizes = inputfile_all_pot_sizes.entries.begin();
    for (auto it_default_sizes = inputfile_default_sizes.entries.begin();
         it_default_sizes != inputfile_default_sizes.entries.end();
         ++it_default_sizes)
    {
      if (it_default_sizes->product_size == product_size) {
        continue;
      }
      product_size = it_default_sizes->product_size;
      while (it_all_pot_sizes != inputfile_all_pot_sizes.entries.end() &&
             it_all_pot_sizes->product_size != product_size)
      {
        ++it_all_pot_sizes;
      }
      if (it_all_pot_sizes == inputfile_all_pot_sizes.entries.end()) {
        break;
      }
      uint16_t best_pot_block_size = 0;
      float best_pot_gflops = 0;
      for (auto it = it_all_pot_sizes;
           it != inputfile_all_pot_sizes.entries.end() && it->product_size == product_size;
           ++it)
      {
        if (it->gflops > best_pot_gflops) {
          best_pot_gflops = it->gflops;
          best_pot_block_size = it->pot_block_size;
        }
      }
      results_entry_t entry;
      entry.product_size = product_size;
      entry.default_block_size = it_default_sizes->nonpot_block_size;
      entry.best_pot_block_size = best_pot_block_size;
      entry.default_gflops = it_default_sizes->gflops;
      entry.best_pot_gflops = best_pot_gflops;
      entry.default_efficiency = entry.default_gflops / entry.best_pot_gflops;
      results.push_back(entry);

      size_triple_t t(product_size);
      if (t.k == t.m && t.m == t.n) {
        cubic_results.push_back(entry);
      }
    }

    cout << "All results:" << endl;
    for (auto it = results.begin(); it != results.end(); ++it) {
      cout << *it << endl;
    }
    cout << endl;

    sort(results.begin(), results.end(), lower_efficiency);
    
    const size_t n = min<size_t>(20, results.size());
    cout << n << " worst results:" << endl;
    for (size_t i = 0; i < n; i++) {
      cout << results[i] << endl;
    }
    cout << endl;

    cout << "cubic results:" << endl;
    for (auto it = cubic_results.begin(); it != cubic_results.end(); ++it) {
      cout << *it << endl;
    }
    cout << endl;

    sort(cubic_results.begin(), cubic_results.end(), lower_efficiency);
    
    cout.precision(2);
    vector<float> a = {0.5f, 0.20f, 0.10f, 0.05f, 0.02f, 0.01f};
    for (auto it = a.begin(); it != a.end(); ++it) {
      size_t n = min(results.size() - 1, size_t(*it * results.size()));
      cout << (100.0f * n / (results.size() - 1))
           << " % of product sizes have default efficiency <= "
           << 100.0f * results[n].default_efficiency << " %" << endl;
    }
    cout.precision(default_precision);
  }
};


void show_usage_and_exit(int argc, char* argv[],
                         const vector<unique_ptr<action_t>>& available_actions)
{
  cerr << "usage: " << argv[0] << " <action> [options...] <input files...>" << endl;
  cerr << "available actions:" << endl;
  for (auto it = available_actions.begin(); it != available_actions.end(); ++it) {
    cerr << "  " << (*it)->invokation_name() << endl;
  } 
  cerr << "the input files should each contain an output of benchmark-blocking-sizes" << endl;
  exit(1);
}

int main(int argc, char* argv[])
{
  cout.precision(default_precision);
  cerr.precision(default_precision);

  vector<unique_ptr<action_t>> available_actions;
  available_actions.emplace_back(new partition_action_t);
  available_actions.emplace_back(new evaluate_defaults_action_t);

  vector<string> input_filenames;

  action_t* action = nullptr;

  if (argc < 2) {
    show_usage_and_exit(argc, argv, available_actions);
  }
  for (int i = 1; i < argc; i++) {
    bool arg_handled = false;
    // Step 1. Try to match action invokation names.
    for (auto it = available_actions.begin(); it != available_actions.end(); ++it) {
      if (!strcmp(argv[i], (*it)->invokation_name())) {
        if (!action) {
          action = it->get();
          arg_handled = true;
          break;
        } else {
          cerr << "can't specify more than one action!" << endl;
          show_usage_and_exit(argc, argv, available_actions);
        }
      }
    }
    if (arg_handled) {
      continue;
    }
    // Step 2. Try to match option names.
    if (argv[i][0] == '-') {
      if (!strcmp(argv[i], "--only-cubic-sizes")) {
        only_cubic_sizes = true;
        arg_handled = true;
      }
      if (!strcmp(argv[i], "--dump-tables")) {
        dump_tables = true;
        arg_handled = true;
      }
      if (!arg_handled) {
        cerr << "Unrecognized option: " << argv[i] << endl;
        show_usage_and_exit(argc, argv, available_actions);
      }
    }
    if (arg_handled) {
      continue;
    }
    // Step 3. Default to interpreting args as input filenames.
    input_filenames.emplace_back(argv[i]);
  }

  if (dump_tables && only_cubic_sizes) {
    cerr << "Incompatible options: --only-cubic-sizes and --dump-tables." << endl;
    show_usage_and_exit(argc, argv, available_actions);
  }

  if (!action) {
    show_usage_and_exit(argc, argv, available_actions);
  }

  action->run(input_filenames);
}
