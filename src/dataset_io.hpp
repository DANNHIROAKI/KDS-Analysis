#pragma once

#include "baselines/baseline_api.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

namespace kexp {

constexpr char kDatasetMagic[8] = {'S', 'J', 'S', 'B', 'O', 'X', '1', '\0'};
constexpr sjs::u32 kDatasetVersion = 1;
constexpr sjs::u32 kScalarTypeFloat32 = 1;

struct DatasetHeader {
  char magic[8];
  sjs::u32 version = 0;
  sjs::u32 scalar_type = 0;
  sjs::u32 dim = 0;
  sjs::u32 reserved = 0;
  sjs::u64 nR = 0;
  sjs::u64 nS = 0;
};

inline bool ReadHeader(const std::string& path, DatasetHeader* out, std::string* err) {
  if (out == nullptr) {
    if (err) *err = "ReadHeader: out is null";
    return false;
  }
  std::ifstream fin(path, std::ios::binary);
  if (!fin) {
    if (err) *err = "ReadHeader: failed to open dataset file: " + path;
    return false;
  }
  fin.read(reinterpret_cast<char*>(out), sizeof(DatasetHeader));
  if (!fin) {
    if (err) *err = "ReadHeader: failed to read dataset header";
    return false;
  }
  if (std::memcmp(out->magic, kDatasetMagic, sizeof(kDatasetMagic)) != 0) {
    if (err) *err = "ReadHeader: bad dataset magic";
    return false;
  }
  if (out->version != kDatasetVersion) {
    if (err) *err = "ReadHeader: unsupported dataset version";
    return false;
  }
  if (out->scalar_type != kScalarTypeFloat32) {
    if (err) *err = "ReadHeader: only float32 datasets are supported";
    return false;
  }
  return true;
}

template <int Dim>
bool LoadBinaryDataset(const std::string& path, sjs::Dataset<Dim, float>* out, std::string* err) {
  if (out == nullptr) {
    if (err) *err = "LoadBinaryDataset: out is null";
    return false;
  }
  DatasetHeader hdr{};
  if (!ReadHeader(path, &hdr, err)) return false;
  if (hdr.dim != static_cast<sjs::u32>(Dim)) {
    if (err) *err = "LoadBinaryDataset: dimension mismatch";
    return false;
  }
  if (hdr.nR > std::numeric_limits<sjs::u32>::max() || hdr.nS > std::numeric_limits<sjs::u32>::max()) {
    if (err) *err = "LoadBinaryDataset: nR/nS exceed u32 payload range";
    return false;
  }

  std::ifstream fin(path, std::ios::binary);
  if (!fin) {
    if (err) *err = "LoadBinaryDataset: failed to open dataset file";
    return false;
  }
  fin.seekg(static_cast<std::streamoff>(sizeof(DatasetHeader)), std::ios::beg);

  auto read_relation = [&](sjs::Relation<Dim, float>* rel, sjs::u64 n) -> bool {
    rel->boxes.resize(static_cast<sjs::usize>(n));
    rel->ids.resize(static_cast<sjs::usize>(n));
    std::array<float, 2 * Dim> row{};
    for (sjs::u64 i = 0; i < n; ++i) {
      fin.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(sizeof(float) * row.size()));
      if (!fin) {
        if (err) *err = "LoadBinaryDataset: failed while reading boxes";
        return false;
      }
      auto& b = rel->boxes[static_cast<sjs::usize>(i)];
      for (int axis = 0; axis < Dim; ++axis) {
        b.lo.v[static_cast<sjs::usize>(axis)] = row[static_cast<sjs::usize>(axis)];
        b.hi.v[static_cast<sjs::usize>(axis)] = row[static_cast<sjs::usize>(Dim + axis)];
      }
      rel->ids[static_cast<sjs::usize>(i)] = static_cast<sjs::Id>(i);
    }
    return true;
  };

  out->R.boxes.clear();
  out->R.ids.clear();
  out->S.boxes.clear();
  out->S.ids.clear();

  if (!read_relation(&out->R, hdr.nR)) return false;
  if (!read_relation(&out->S, hdr.nS)) return false;
  return true;
}

}  // namespace kexp
