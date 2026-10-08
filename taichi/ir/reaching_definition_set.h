#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace taichi::lang {

class Stmt;

// Experimental representation switch: one index per analysis, one bit per
// definition per node. The hash representation remains the control.
struct ReachingDefinitionUniverse {
  std::vector<Stmt *> values;
  std::unordered_map<Stmt *, std::size_t> indices;
};

class ReachingDefinitionSet {
 public:
  using HashSet = std::unordered_set<Stmt *>;
  using Universe = std::shared_ptr<const ReachingDefinitionUniverse>;

  class Iterator {
   public:
    Iterator(const ReachingDefinitionSet *owner, std::size_t index)
        : owner_(owner), index_(index) {}
    Iterator(const ReachingDefinitionSet *owner, HashSet::const_iterator it)
        : owner_(owner), hash_it_(it) {}
    Stmt *const &operator*() const {
      return owner_->universe_ ? owner_->universe_->values[index_] : *hash_it_;
    }
    Iterator &operator++() {
      if (owner_->universe_) {
        index_ = owner_->next_set_bit(index_ + 1);
      } else {
        ++hash_it_;
      }
      return *this;
    }
    bool operator==(const Iterator &other) const {
      return owner_->universe_ ? index_ == other.index_
                              : hash_it_ == other.hash_it_;
    }
    bool operator!=(const Iterator &other) const { return !(*this == other); }

   private:
    const ReachingDefinitionSet *owner_;
    std::size_t index_{0};
    HashSet::const_iterator hash_it_;
  };

  ReachingDefinitionSet() = default;
  ReachingDefinitionSet(const ReachingDefinitionSet &) = default;
  ReachingDefinitionSet &operator=(const ReachingDefinitionSet &) = default;
  // The worklist moves out old_out before assigning gen to the original set.
  // Preserve the universe of the moved-from object for that assignment.
  ReachingDefinitionSet(ReachingDefinitionSet &&other) noexcept
      : universe_(other.universe_),
        words_(std::move(other.words_)),
        hash_(std::move(other.hash_)) {}

  void reset(Universe universe) {
    universe_ = std::move(universe);
    words_.assign(universe_ ? (universe_->values.size() + 63) / 64 : 0, 0);
    HashSet().swap(hash_);
  }
  void clear() {
    if (universe_) {
      std::fill(words_.begin(), words_.end(), 0);
    } else {
      hash_.clear();
    }
  }
  ReachingDefinitionSet &operator=(const HashSet &values) {
    if (universe_) {
      words_.assign((universe_->values.size() + 63) / 64, 0);
      for (auto value : values) insert(value);
    } else {
      hash_ = values;
    }
    return *this;
  }
  void insert(Stmt *value) {
    if (universe_) {
      auto index = universe_->indices.at(value);
      words_[index / 64] |= uint64_t{1} << (index % 64);
    } else {
      hash_.insert(value);
    }
  }
  void unite(const ReachingDefinitionSet &other) {
    if (universe_) {
      for (std::size_t i = 0; i < words_.size(); ++i) words_[i] |= other.words_[i];
    } else {
      hash_.insert(other.hash_.begin(), other.hash_.end());
    }
  }
  void unite_without(const ReachingDefinitionSet &other,
                     const ReachingDefinitionSet &removed) {
    if (universe_) {
      for (std::size_t i = 0; i < words_.size(); ++i) {
        words_[i] |= other.words_[i] & ~removed.words_[i];
      }
    } else {
      for (auto value : other.hash_) {
        if (!removed.hash_.count(value)) hash_.insert(value);
      }
    }
  }
  Iterator begin() const {
    return universe_ ? Iterator(this, next_set_bit(0)) : Iterator(this, hash_.begin());
  }
  Iterator end() const {
    return universe_ ? Iterator(this, universe_->values.size()) : Iterator(this, hash_.end());
  }
  Iterator find(Stmt *value) const {
    if (!universe_) return Iterator(this, hash_.find(value));
    auto it = universe_->indices.find(value);
    if (it == universe_->indices.end()) return end();
    auto index = it->second;
    return (words_[index / 64] & (uint64_t{1} << (index % 64)))
               ? Iterator(this, index) : end();
  }
  bool empty() const { return begin() == end(); }
  const Universe &universe() const { return universe_; }
  bool operator!=(const ReachingDefinitionSet &other) const {
    return universe_ ? words_ != other.words_ : hash_ != other.hash_;
  }

 private:
  std::size_t next_set_bit(std::size_t index) const {
    const auto count = universe_->values.size();
    if (index >= count) return count;
    auto word_index = index / 64;
    auto word = words_[word_index] & (~uint64_t{0} << (index % 64));
    while (!word) {
      if (++word_index == words_.size()) return count;
      word = words_[word_index];
    }
    return word_index * 64 + __builtin_ctzll(word);
  }
  Universe universe_;
  std::vector<uint64_t> words_;
  HashSet hash_;
};

}  // namespace taichi::lang
