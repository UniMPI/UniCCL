/* unicc_dtmap.c - per-backend datatype / reduce-op mapping tables.
 *
 * The tables are global and fixed after unicc_vtable_init; the collective hot
 * path in unicc_api.c reads them by array index (no switch, no backend branch),
 * matching UniMPI's steady-state cost model. See include/unicc_dtmap.h for the
 * design rationale. */

#include "unicc_dtmap.h"

int unicc_dt_to_native[UNICC_DT_COUNT] = {0};
int unicc_op_to_native[UNICC_OP_COUNT] = {0};

void unicc_dtmap_reset_nccl(void) {
    /* Modern NCCL numbering (also oneCCL v2, RCCL, Hygon DCU):
     * ncclInt8=0 Uint8=1 Int32=2 Uint32=3 Int64=4 Uint64=5 Float16=6
     * Float32=7 Float64=8 Bfloat16=9  (indexed by unicc_datatype_t). */
    unicc_dt_to_native[0] = 0;  /* UNICC_I8  -> ncclInt8      */
    unicc_dt_to_native[1] = 1;  /* UNICC_U8  -> ncclUint8     */
    unicc_dt_to_native[2] = 2;  /* UNICC_I32 -> ncclInt32     */
    unicc_dt_to_native[3] = 3;  /* UNICC_U32 -> ncclUint32    */
    unicc_dt_to_native[4] = 4;  /* UNICC_I64 -> ncclInt64     */
    unicc_dt_to_native[5] = 5;  /* UNICC_U64 -> ncclUint64    */
    unicc_dt_to_native[6] = 6;  /* UNICC_F16 -> ncclFloat16   */
    unicc_dt_to_native[7] = 7;  /* UNICC_F32 -> ncclFloat32   */
    unicc_dt_to_native[8] = 8;  /* UNICC_F64 -> ncclFloat64   */
    unicc_dt_to_native[9] = 9;  /* UNICC_BF16-> ncclBfloat16  */

    /* ncclSum=0 Prod=1 Max=2 Min=3 (indexed by unicc_reduce_op_t). */
    unicc_op_to_native[0] = 0;  /* UNICC_SUM  */
    unicc_op_to_native[1] = 1;  /* UNICC_PROD */
    unicc_op_to_native[2] = 2;  /* UNICC_MAX  */
    unicc_op_to_native[3] = 3;  /* UNICC_MIN  */
}

int unicc_dtmap_lookup_dt(int dt) {
    if (dt < 0 || dt >= UNICC_DT_COUNT) {
        return -1;
    }
    return unicc_dt_to_native[dt];
}

int unicc_dtmap_lookup_op(int op) {
    if (op < 0 || op >= UNICC_OP_COUNT) {
        return -1;
    }
    return unicc_op_to_native[op];
}
