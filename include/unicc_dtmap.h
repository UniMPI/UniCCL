#ifndef UNICC_DTMAP_H
#define UNICC_DTMAP_H

/* Datatype / reduce-op mapping (unicc_datatype_t / unicc_reduce_op_t ->
 * backend-native numeric values).
 *
 * Every current backend (NCCL, RCCL, oneCCL v2, ECCL) uses the NCCL-family
 * numbering, so the mapping is the identity and lives here as header-static
 * accessors - the collective hot path (unicc_api.c) inlines them to a single
 * bounds-checked load.  There is deliberately no table cell to overwrite and no
 * init-time fill: a future P2 backend that diverges (Moore/MetaX MCCL with a
 * Uint32 inserted, Cambricon CNCL hex coding, Ascend HCCL layout - see
 * docs/BACKENDS.md "Enum mapping") grows a small per-family table behind these
 * same accessors, hot path unchanged.
 *
 * Historically this was src/unicc_dtmap.c with mutable globals filled at
 * unicc_vtable_init; the mutable-global machinery was removed as dead weight
 * once no backend diverged (F8/F10 review 2026-09).
 */

#define UNICC_DT_COUNT 10
#define UNICC_OP_COUNT 4

/* Native (backend) value for each UniCCL datatype/op, or -1 when out of range.
 * Current backends are identity: UNICC_I8=0 .. UNICC_BF16=9,
 * UNICC_SUM=0 .. UNICC_MIN=3 (NCCL / oneCCL v2 numbering). */
static inline int unicc_dtmap_lookup_dt(int dt) {
    return (dt >= 0 && dt < UNICC_DT_COUNT) ? dt : -1;
}

static inline int unicc_dtmap_lookup_op(int op) {
    return (op >= 0 && op < UNICC_OP_COUNT) ? op : -1;
}

#endif /* UNICC_DTMAP_H */
