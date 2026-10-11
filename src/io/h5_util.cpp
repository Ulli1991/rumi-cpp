#include "h5_util.h"

namespace h5 {

namespace {
struct Silence {   // HDF5 prints its error stack by default; we report through exceptions instead
    Silence() { H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr); }
};
const Silence silence_once;

hid_t vlen_utf8() {
    hid_t t = H5Tcopy(H5T_C_S1);
    H5Tset_size(t, H5T_VARIABLE);
    H5Tset_cset(t, H5T_CSET_UTF8);
    return t;
}

Handle make_attr(hid_t loc, const std::string& name, hid_t ftype, hid_t space) {
    if (H5Aexists(loc, name.c_str()) > 0) H5Adelete(loc, name.c_str());
    hid_t a = H5Acreate2(loc, name.c_str(), ftype, space, H5P_DEFAULT, H5P_DEFAULT);
    if (a < 0) fail("cannot create attribute " + name);
    return Handle(a, H5Aclose);
}

Handle open_attr(hid_t loc, const std::string& name) {
    hid_t a = H5Aopen(loc, name.c_str(), H5P_DEFAULT);
    if (a < 0) fail("missing attribute " + name);
    return Handle(a, H5Aclose);
}
}  // namespace

void fail(const std::string& what) { throw std::runtime_error("HDF5: " + what); }

Handle create_file(const std::string& path) {
    hid_t f = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (f < 0) fail("cannot create " + path);
    return Handle(f, H5Fclose);
}

Handle open_file(const std::string& path) {
    hid_t f = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) fail("cannot open " + path);
    return Handle(f, H5Fclose);
}

Handle create_group(hid_t loc, const std::string& name) {
    hid_t g = H5Gcreate2(loc, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (g < 0) fail("cannot create group " + name);
    return Handle(g, H5Gclose);
}

Handle open_group(hid_t loc, const std::string& name) {
    hid_t g = H5Gopen2(loc, name.c_str(), H5P_DEFAULT);
    if (g < 0) fail("missing group " + name);
    return Handle(g, H5Gclose);
}

bool exists(hid_t loc, const std::string& name) { return H5Lexists(loc, name.c_str(), H5P_DEFAULT) > 0; }
bool has_attr(hid_t loc, const std::string& name) { return H5Aexists(loc, name.c_str()) > 0; }

template <> hid_t mem_type<double>() { return H5T_NATIVE_DOUBLE; }
template <> hid_t mem_type<int>() { return H5T_NATIVE_INT; }
template <> hid_t mem_type<int64_t>() { return H5T_NATIVE_INT64; }
template <> hid_t mem_type<uint64_t>() { return H5T_NATIVE_UINT64; }
template <> hid_t mem_type<uint8_t>() { return H5T_NATIVE_UINT8; }
template <> hid_t mem_type<char>() { return H5T_NATIVE_CHAR; }
template <> hid_t file_type<double>() { return H5T_IEEE_F64LE; }
template <> hid_t file_type<int>() { return H5T_STD_I32LE; }
template <> hid_t file_type<int64_t>() { return H5T_STD_I64LE; }
template <> hid_t file_type<uint64_t>() { return H5T_STD_U64LE; }
template <> hid_t file_type<uint8_t>() { return H5T_STD_U8LE; }
template <> hid_t file_type<char>() { return H5T_STD_I8LE; }

template <class T>
void write(hid_t loc, const std::string& name, const T* data, size_t n, size_t ncol) {
    hsize_t dims[2] = {(hsize_t)n, (hsize_t)ncol};
    Handle space(H5Screate_simple(ncol ? 2 : 1, dims, nullptr), H5Sclose);
    if (space < 0) fail("dataspace of " + name);
    Handle ds(H5Dcreate2(loc, name.c_str(), file_type<T>(), space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
    if (ds < 0) fail("cannot create dataset " + name);
    if (n * (ncol ? ncol : 1) > 0 && H5Dwrite(ds, mem_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0)
        fail("cannot write dataset " + name);
}

template <class T>
size_t read(hid_t loc, const std::string& name, std::vector<T>& v) {
    Handle ds(H5Dopen2(loc, name.c_str(), H5P_DEFAULT), H5Dclose);
    if (ds < 0) fail("missing dataset " + name);
    Handle space(H5Dget_space(ds), H5Sclose);
    int rank = H5Sget_simple_extent_ndims(space);
    hsize_t dims[8] = {0};
    if (rank < 0 || rank > 8) fail("bad rank of " + name);
    H5Sget_simple_extent_dims(space, dims, nullptr);
    size_t total = 1;
    for (int k = 0; k < rank; k++) total *= dims[k];
    v.resize(total);
    if (total > 0 && H5Dread(ds, mem_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data()) < 0)
        fail("cannot read dataset " + name);
    return rank > 0 ? (size_t)dims[0] : 1;
}

#define H5_INSTANTIATE(T) \
    template void write<T>(hid_t, const std::string&, const T*, size_t, size_t); \
    template size_t read<T>(hid_t, const std::string&, std::vector<T>&);
H5_INSTANTIATE(double)
H5_INSTANTIATE(int)
H5_INSTANTIATE(int64_t)
H5_INSTANTIATE(uint64_t)
H5_INSTANTIATE(uint8_t)
H5_INSTANTIATE(char)
#undef H5_INSTANTIATE

namespace {
template <class T>
void scalar_attr(hid_t loc, const std::string& name, T v) {
    Handle space(H5Screate(H5S_SCALAR), H5Sclose);
    Handle a = make_attr(loc, name, file_type<T>(), space);
    if (H5Awrite(a, mem_type<T>(), &v) < 0) fail("cannot write attribute " + name);
}
template <class T>
void array_attr(hid_t loc, const std::string& name, const std::vector<T>& v) {
    hsize_t n = v.size();
    Handle space(H5Screate_simple(1, &n, nullptr), H5Sclose);
    Handle a = make_attr(loc, name, file_type<T>(), space);
    if (n > 0 && H5Awrite(a, mem_type<T>(), v.data()) < 0) fail("cannot write attribute " + name);
}
}  // namespace

void attr(hid_t loc, const std::string& name, double v) { scalar_attr<double>(loc, name, v); }
void attr(hid_t loc, const std::string& name, int v) { scalar_attr<int>(loc, name, v); }
void attr(hid_t loc, const std::string& name, long long v) { scalar_attr<int64_t>(loc, name, (int64_t)v); }
void attr(hid_t loc, const std::string& name, const std::vector<double>& v) { array_attr<double>(loc, name, v); }
void attr(hid_t loc, const std::string& name, const std::vector<long long>& v) {
    std::vector<int64_t> w(v.begin(), v.end());
    array_attr<int64_t>(loc, name, w);
}

void attr(hid_t loc, const std::string& name, const std::string& v) {
    Handle type(vlen_utf8(), H5Tclose);
    Handle space(H5Screate(H5S_SCALAR), H5Sclose);
    Handle a = make_attr(loc, name, type, space);
    const char* p = v.c_str();
    if (H5Awrite(a, type, &p) < 0) fail("cannot write attribute " + name);
}

double get_double(hid_t loc, const std::string& name) {
    Handle a = open_attr(loc, name);
    double v = 0;
    if (H5Aread(a, H5T_NATIVE_DOUBLE, &v) < 0) fail("cannot read attribute " + name);
    return v;
}

long long get_int(hid_t loc, const std::string& name) {
    Handle a = open_attr(loc, name);
    int64_t v = 0;
    if (H5Aread(a, H5T_NATIVE_INT64, &v) < 0) fail("cannot read attribute " + name);
    return v;
}

std::vector<double> get_doubles(hid_t loc, const std::string& name) {
    Handle a = open_attr(loc, name);
    Handle space(H5Aget_space(a), H5Sclose);
    hssize_t n = H5Sget_simple_extent_npoints(space);
    std::vector<double> v(n > 0 ? n : 0);
    if (n > 0 && H5Aread(a, H5T_NATIVE_DOUBLE, v.data()) < 0) fail("cannot read attribute " + name);
    return v;
}

std::string get_string(hid_t loc, const std::string& name) {
    Handle a = open_attr(loc, name);
    Handle ftype(H5Aget_type(a), H5Tclose);
    if (H5Tis_variable_str(ftype) > 0) {
        Handle type(vlen_utf8(), H5Tclose);
        char* p = nullptr;
        if (H5Aread(a, type, &p) < 0) fail("cannot read attribute " + name);
        std::string s = p ? p : "";
        if (p) H5free_memory(p);
        return s;
    }
    size_t n = H5Tget_size(ftype);
    std::string s(n, '\0');
    Handle type(H5Tcopy(H5T_C_S1), H5Tclose);
    H5Tset_size(type, n);
    if (H5Aread(a, type, s.data()) < 0) fail("cannot read attribute " + name);
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

}  // namespace h5
