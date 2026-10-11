#pragma once
// Thin RAII layer over the HDF5 C API for the restart and snapshot files (io/restart.cpp, io/snapshot.cpp).
// Every failure throws std::runtime_error with the object name; doubles are stored as IEEE little-endian
// 64-bit, so a write and a read give back the same bits.

#include <hdf5.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace h5 {

struct Handle {
    hid_t id = -1;
    herr_t (*closer)(hid_t) = nullptr;
    Handle() = default;
    Handle(hid_t i, herr_t (*c)(hid_t)) : id(i), closer(c) {}
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& o) noexcept : id(o.id), closer(o.closer) { o.id = -1; }
    Handle& operator=(Handle&& o) noexcept {
        if (this != &o) { reset(); id = o.id; closer = o.closer; o.id = -1; }
        return *this;
    }
    ~Handle() { reset(); }
    void reset() { if (id >= 0 && closer) closer(id); id = -1; }
    operator hid_t() const { return id; }
};

[[noreturn]] void fail(const std::string& what);

Handle create_file(const std::string& path);         // truncates
Handle open_file(const std::string& path);           // read-only
Handle create_group(hid_t loc, const std::string& name);
Handle open_group(hid_t loc, const std::string& name);
bool exists(hid_t loc, const std::string& name);     // link (group or dataset) exists
bool has_attr(hid_t loc, const std::string& name);

template <class T> hid_t mem_type();                 // native type of T
template <class T> hid_t file_type();                // fixed little-endian type of T

// Dataset of n rows (ncol = 0: 1-d, else n x ncol, row-major)
template <class T>
void write(hid_t loc, const std::string& name, const T* data, size_t n, size_t ncol = 0);
template <class T>
void write(hid_t loc, const std::string& name, const std::vector<T>& v, size_t ncol = 0) {
    write(loc, name, v.data(), ncol ? v.size() / ncol : v.size(), ncol);
}
// Read a dataset (any rank) into v; returns its first dimension. Element count must match T's storage size.
template <class T>
size_t read(hid_t loc, const std::string& name, std::vector<T>& v);

void attr(hid_t loc, const std::string& name, double v);
void attr(hid_t loc, const std::string& name, int v);
void attr(hid_t loc, const std::string& name, long long v);
void attr(hid_t loc, const std::string& name, const std::string& v);
void attr(hid_t loc, const std::string& name, const std::vector<double>& v);
void attr(hid_t loc, const std::string& name, const std::vector<long long>& v);
inline void attr(hid_t loc, const std::string& name, const char* v) { attr(loc, name, std::string(v)); }
inline void attr(hid_t loc, const std::string& name, bool v) { attr(loc, name, (int)v); }

double get_double(hid_t loc, const std::string& name);
long long get_int(hid_t loc, const std::string& name);
std::string get_string(hid_t loc, const std::string& name);
std::vector<double> get_doubles(hid_t loc, const std::string& name);

}  // namespace h5
