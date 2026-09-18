#ifndef UNICC_DTMAP_H
#define UNICC_DTMAP_H

/* Per-backend datatype / reduce-op mapping tables.
 *
 * Vendors disagree on the numeric values of the datatype enum (NCCL and oneCCL
 * v2 use one numbering, MCCL (Moore/MetaX) uses the RCCL-style numbering with
 * Uint32 inserted, Cambricon CNCL uses a hex coding, Ascend HCCL another
 * layout entirely), so UniCCL maps its own unicc_datatype_t / unicc_reduce_op_t
 * onto the active backend's native values through tables that are filled once
 * at initialization. The collective hot path then uses plain array indexing -
 * no switch, no branch on backend (see docs/BACKENDS.md "Enum mapping").
 *
 * Table mechanics mirror UniMPI's per-backend predefined-constant
 * initialization (init_<backend>_error_codes): unicc_vtable_init resets the
 * tables to the NCCL-family values first, then the matching backend binding
 * overwrites whatever differs. Indexing uses the UniCCL enum values directly
 * (UNICC_I8..UNICC_BF16 are 0..9, UNICC_SUM..UNICC_MIN are 0..3).
 */

#define UNICC_DT_COUNT 10
#define UNICC_OP_COUNT 4

/* Native(backend) value for each UniCCL datatype/op, or -1 when the backend
 * has no such type/op. Filled by unicc_dtmap_reset_nccl() plus per-backend
 * setters. Indexed by unicc_datatype_t / unicc_reduce_op_t. */
extern int unicc_dt_to_native[UNICC_DT_COUNT];
extern int unicc_op_to_native[UNICC_OP_COUNT];

/* Reset both tables to the NCCL-family numbering:
 *   I8=0 U8=1 I32=2 U32=3 I64=4 U64=5 F16=6 F32=7 F64=8 BF16=9;
 *   SUM=0 PROD=1 MAX=2 MIN=3. (Modern NCCL / oneCCL v2 values; historically
 *   remote versions of this header used the older NCCL numbering F16=9,
 *   BF16=10 — see docs/BACKENDS.md and the version notes.) */
void unicc_dtmap_reset_nccl(void);

/* Per-backend overwrites (declared here for tests; implemented in the backend
 * bindings). P1: oneCCL v2 and ECCL keep the NCCL-family numbering, so no
 * overwrite is needed. Differing backends (CNCL/HCCL/MCCL) land in P2 and add
 * their setters. */
int unicc_dtmap_lookup_dt(int dt);   /* dt index -> native value or -1 */
int unicc_dtmap_lookup_op(int op);   /* op index -> native value or -1 */

#endif /* UNICC_DTMAP_H */
