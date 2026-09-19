#ifndef UNICC_BIND_H
#define UNICC_BIND_H

#include "unicc_platform.h"
#include "unicc_backends.h"

/* Generic family binder (unicc_bind.c).
 *
 * One implementation serves every backend family; the per-family facts (symbol
 * prefix, bootstrap-id size, signature quirks) live in the unicc_backends[]
 * descriptor selected by type. Each unicc_vtable_init_<family>() is now a
 * one-liner that calls this with its family type (see the four per-family
 * backends, e.g. nccl.c).
 *
 * The DESCRIPTOR's own family is the only one bound and the only one
 * validated: core slots (GetVersion / CommInitRank / AllReduce / Broadcast)
 * must resolve for that family or UNICC_ERR_SYMBOL_NOT_FOUND is returned
 * (core missing => unicc_init fails, per docs/BACKENDS.md). Optional slots
 * degrade to a NULL vtable slot; a wrapper is installed only over a resolved
 * symbol - never a wrapper whose inner call would be NULL. */
int unicc_vtable_bind(unicc_lib_handle_t handle, unicc_backend_type_t type);

#endif /* UNICC_BIND_H */
