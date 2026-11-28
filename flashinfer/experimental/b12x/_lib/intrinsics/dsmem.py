# Copyright (c) 2026 by the b12x authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SM120 Distributed Shared Memory (DSMEM) cluster load intrinsics."""

from __future__ import annotations

from typing import Tuple

from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def ld_dsm_f32(smem_addr: Int32, peer_rank: Int32, *, loc=None, ip=None) -> Float32:
    """Load one FP32 scalar from a peer CTA's shared memory via DSMEM."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Int32(smem_addr).ir_value(loc=loc, ip=ip),
                Int32(peer_rank).ir_value(loc=loc, ip=ip),
            ],
            """{
                .reg .u32 raddr;
                mapa.shared::cluster.u32 raddr, $1, $2;
                ld.shared::cluster.f32 $0, [raddr];
            }""",
            "=f,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_dsm_v4_u32(
    smem_addr: Int32, peer_rank: Int32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load one 128-bit u32x4 vector from a peer CTA's shared memory via DSMEM."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int32(peer_rank).ir_value(loc=loc, ip=ip),
        ],
        """{
            .reg .u32 raddr;
            mapa.shared::cluster.u32 raddr, $4, $5;
            ld.shared::cluster.v4.u32 {$0, $1, $2, $3}, [raddr];
        }""",
        "=r,=r,=r,=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def ld_dsm_v4_f32(
    smem_addr: Int32, peer_rank: Int32, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Load one 128-bit f32x4 vector from a peer CTA's shared memory via DSMEM."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int32(peer_rank).ir_value(loc=loc, ip=ip),
        ],
        """{
            .reg .u32 raddr;
            mapa.shared::cluster.u32 raddr, $4, $5;
            ld.shared::cluster.v4.f32 {$0, $1, $2, $3}, [raddr];
        }""",
        "=f,=f,=f,=f,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [3], loc=loc, ip=ip)),
    )


__all__ = ["ld_dsm_f32", "ld_dsm_v4_f32", "ld_dsm_v4_u32"]
