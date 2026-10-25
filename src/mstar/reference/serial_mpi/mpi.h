// RUMI++ MSTAR module (GPLv3, see src/mstar/LICENSE)
// Copyright (C) 2026 Ulrich Steinwandel.
// Licensed under the GPLv3 license (or, at your option, any later version).
//
// Single-task stand-in for the subset of MPI that the KETJU integrator library uses, so that the vendored library
// (src/mstar/ketju) builds and runs without an MPI installation (RUMI++ builds with USE_MPI=OFF). Every
// communicator has exactly one task; collectives with MPI_IN_PLACE are no-ops, and the non-in-place reduction copies
// its input. Not a general MPI replacement.

#ifndef RUMI_MSTAR_SERIAL_MPI_H
#define RUMI_MSTAR_SERIAL_MPI_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int MPI_Comm;
typedef int MPI_Group;
typedef int MPI_Datatype;
typedef int MPI_Op;
typedef int MPI_Request;
typedef struct { int dummy; } MPI_Status;

#define MPI_COMM_NULL 0
#define MPI_COMM_WORLD 1
#define MPI_COMM_SELF 2
#define MPI_STATUSES_IGNORE ((MPI_Status *)0)
#define MPI_IN_PLACE ((void *)1)
#define MPI_SUCCESS 0

#define MPI_DATATYPE_NULL 0
#define MPI_DOUBLE 1
#define MPI_INT 2
#define MPI_DOUBLE_INT 3
#define MPI_C_BOOL 4

#define MPI_SUM 1
#define MPI_MAX 2
#define MPI_MIN 3
#define MPI_MINLOC 4
#define MPI_LOR 5

struct rumi_serial_mpi_double_int { double d; int i; };
#ifdef __cplusplus
#define RUMI_SERIAL_MPI_BOOL bool
#else
#define RUMI_SERIAL_MPI_BOOL _Bool
#endif

static inline size_t rumi_serial_mpi_type_size(MPI_Datatype t) {
    switch (t) {
    case MPI_DOUBLE: return sizeof(double);
    case MPI_INT: return sizeof(int);
    case MPI_DOUBLE_INT: return sizeof(struct rumi_serial_mpi_double_int);
    case MPI_C_BOOL: return sizeof(RUMI_SERIAL_MPI_BOOL);
    default: return 0;
    }
}

static inline int MPI_Init(int *argc, char ***argv) { (void)argc; (void)argv; return 0; }
static inline int MPI_Finalize(void) { return 0; }
static inline int MPI_Comm_size(MPI_Comm c, int *s) { (void)c; *s = 1; return 0; }
static inline int MPI_Comm_rank(MPI_Comm c, int *r) { (void)c; *r = 0; return 0; }
static inline int MPI_Comm_dup(MPI_Comm c, MPI_Comm *n) { (void)c; *n = MPI_COMM_SELF; return 0; }
static inline int MPI_Comm_free(MPI_Comm *c) { *c = MPI_COMM_NULL; return 0; }
static inline int MPI_Comm_group(MPI_Comm c, MPI_Group *g) { (void)c; *g = 1; return 0; }
static inline int MPI_Group_range_incl(MPI_Group g, int n, int ranges[][3], MPI_Group *ng) {
    (void)g; (void)n; (void)ranges; *ng = 1; return 0;
}
static inline int MPI_Group_rank(MPI_Group g, int *r) { (void)g; *r = 0; return 0; }
static inline int MPI_Group_size(MPI_Group g, int *s) { (void)g; *s = 1; return 0; }
static inline int MPI_Group_free(MPI_Group *g) { *g = 0; return 0; }
static inline int MPI_Comm_create_group(MPI_Comm c, MPI_Group g, int tag, MPI_Comm *n) {
    (void)c; (void)g; (void)tag; *n = MPI_COMM_SELF; return 0;
}
static inline int MPI_Allreduce(const void *s, void *r, int n, MPI_Datatype t, MPI_Op op, MPI_Comm c) {
    (void)op; (void)c;
    if (s != MPI_IN_PLACE) memcpy(r, s, (size_t)n * rumi_serial_mpi_type_size(t));
    return 0;
}
static inline int MPI_Ibcast(void *b, int n, MPI_Datatype t, int root, MPI_Comm c, MPI_Request *req) {
    (void)b; (void)n; (void)t; (void)root; (void)c; *req = 0; return 0;
}
static inline int MPI_Waitall(int n, MPI_Request *req, MPI_Status *st) { (void)n; (void)req; (void)st; return 0; }
static inline int MPI_Allgatherv(const void *s, int sc, MPI_Datatype st, void *r, const int *rc, const int *displs,
                                 MPI_Datatype rt, MPI_Comm c) {
    (void)sc; (void)st; (void)c;
    if (s != MPI_IN_PLACE) memcpy((char *)r + (size_t)displs[0] * rumi_serial_mpi_type_size(rt), s,
                                  (size_t)rc[0] * rumi_serial_mpi_type_size(rt));
    return 0;
}
static inline int MPI_Abort(MPI_Comm c, int code) { (void)c; fflush(stderr); exit(code); }
static inline double MPI_Wtime(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

#ifdef __cplusplus
}
#endif

#endif // RUMI_MSTAR_SERIAL_MPI_H
