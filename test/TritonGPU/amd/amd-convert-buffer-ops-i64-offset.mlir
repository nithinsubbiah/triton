// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="arch-generation-name=gfx942" | FileCheck %s

// Test that tt.load with i64 offsets derived from provably bounded non-negative
// expressions is converted to amdg.buffer_load with an arith.trunci from i64 to i32.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: @load_i64_offset_bounded
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @load_i64_offset_bounded(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<256xf32, #blocked> {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %range_ext = arith.extsi %range : tensor<256xi32, #blocked> to tensor<256xi64, #blocked>
    %c1024_i64 = arith.constant 1024 : i64
    %stride = tt.splat %c1024_i64 : i64 -> tensor<256xi64, #blocked>
    %offset = arith.muli %range_ext, %stride : tensor<256xi64, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %base, %offset : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi64, #blocked>
    // CHECK: arith.trunci
    // CHECK-SAME: tensor<256xi64,
    // CHECK-SAME: to tensor<256xi32,
    // CHECK: amdg.buffer_load
    %val = tt.load %ptr : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return %val : tensor<256xf32, #blocked>
  }
}

// -----

// Test that i64 offset loads are NOT converted when the offset may be negative.

#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: @load_i64_offset_possibly_negative
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @load_i64_offset_possibly_negative(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i64) -> tensor<256xf32, #blocked1> {
    %splat_off = tt.splat %arg1 : i64 -> tensor<256xi64, #blocked1>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked1>
    %ptr = tt.addptr %base, %splat_off : tensor<256x!tt.ptr<f32>, #blocked1>, tensor<256xi64, #blocked1>
    // CHECK-NOT: amdg.buffer_load
    // CHECK: tt.load
    %val = tt.load %ptr : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return %val : tensor<256xf32, #blocked1>
  }
}

// -----

// Test that i64 offset stores are converted with trunci when offset is bounded.

#blocked2 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: @store_i64_offset_bounded
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @store_i64_offset_bounded(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %data: tensor<256xf32, #blocked2>) {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked2>
    %range_ext = arith.extsi %range : tensor<256xi32, #blocked2> to tensor<256xi64, #blocked2>
    %c512_i64 = arith.constant 512 : i64
    %stride = tt.splat %c512_i64 : i64 -> tensor<256xi64, #blocked2>
    %offset = arith.muli %range_ext, %stride : tensor<256xi64, #blocked2>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked2>
    %ptr = tt.addptr %base, %offset : tensor<256x!tt.ptr<f32>, #blocked2>, tensor<256xi64, #blocked2>
    // CHECK: arith.trunci
    // CHECK-SAME: tensor<256xi64,
    // CHECK-SAME: to tensor<256xi32,
    // CHECK: amdg.buffer_store
    tt.store %ptr, %data : tensor<256x!tt.ptr<f32>, #blocked2>
    tt.return
  }
}

// -----

// Test that i64 offset loads with tt.pointer_range=32 attribute are converted.

#blocked3 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

// CHECK-LABEL: @load_i64_offset_pointer_range_32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @load_i64_offset_pointer_range_32(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: i64) -> tensor<256xf32, #blocked3> {
    %splat_off = tt.splat %arg1 : i64 -> tensor<256xi64, #blocked3>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked3>
    %ptr = tt.addptr %base, %splat_off : tensor<256x!tt.ptr<f32>, #blocked3>, tensor<256xi64, #blocked3>
    // CHECK: arith.trunci
    // CHECK-SAME: tensor<256xi64,
    // CHECK-SAME: to tensor<256xi32,
    // CHECK: amdg.buffer_load
    %val = tt.load %ptr : tensor<256x!tt.ptr<f32>, #blocked3>
    tt.return %val : tensor<256xf32, #blocked3>
  }
}

// -----

// Test that multiple loads sharing the same tt.addptr with i64 offset are both
// converted without SSA dominance violations (regression test for #9907).

#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 1], order = [1, 0]}>

// CHECK-LABEL: @multi_load_shared_addptr_i64
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @multi_load_shared_addptr_i64(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %r = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %ext = arith.extsi %r : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> to tensor<8xi64, #ttg.slice<{dim = 0, parent = #blocked4}>>
    %offset = tt.expand_dims %ext {axis = 0 : i32} : tensor<8xi64, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x8xi64, #blocked4>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>, #blocked4>
    %ptr = tt.addptr %base, %offset : tensor<1x8x!tt.ptr<f32>, #blocked4>, tensor<1x8xi64, #blocked4>
    %v1 = tt.load %ptr : tensor<1x8x!tt.ptr<f32>, #blocked4>
    %v2 = tt.load %ptr : tensor<1x8x!tt.ptr<f32>, #blocked4>
    // trunci ops are inserted right after the offset definition
    // CHECK-COUNT-2: arith.trunci {{.*}} tensor<1x8xi64, {{.*}}> to tensor<1x8xi32, {{.*}}>
    // CHECK: amdg.buffer_load
    // CHECK: amdg.buffer_load
    // CHECK-NOT: tt.load
    %sum = arith.addf %v1, %v2 : tensor<1x8xf32, #blocked4>
    %st = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>, #blocked4>
    tt.store %st, %sum : tensor<1x8x!tt.ptr<f32>, #blocked4>
    tt.return
  }
}

// -----

// Test that a store with i64 offset is correctly converted.

#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 1], order = [1, 0]}>

// CHECK-LABEL: @store_i64_offset
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @store_i64_offset(%arg0: !tt.ptr<f32>, %arg1: tensor<1x8xf32, #blocked5>) {
    %r = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
    %ext = arith.extsi %r : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked5}>> to tensor<8xi64, #ttg.slice<{dim = 0, parent = #blocked5}>>
    %offset = tt.expand_dims %ext {axis = 0 : i32} : tensor<8xi64, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x8xi64, #blocked5>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>, #blocked5>
    %ptr = tt.addptr %base, %offset : tensor<1x8x!tt.ptr<f32>, #blocked5>, tensor<1x8xi64, #blocked5>
    tt.store %ptr, %arg1 : tensor<1x8x!tt.ptr<f32>, #blocked5>
    // CHECK: arith.trunci {{.*}} tensor<1x8xi64, {{.*}}> to tensor<1x8xi32, {{.*}}>
    // CHECK: amdg.buffer_store
    // CHECK-NOT: tt.store
    tt.return
  }
}

// -----

// Test that two stores sharing the same addptr with i64 offset are both
// converted without SSA dominance violations.

#blocked6 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [2, 1], order = [1, 0]}>

// CHECK-LABEL: @multi_store_shared_addptr_i64
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @multi_store_shared_addptr_i64(%arg0: !tt.ptr<f32>, %arg1: tensor<1x8xf32, #blocked6>) {
    %r = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked6}>>
    %ext = arith.extsi %r : tensor<8xi32, #ttg.slice<{dim = 0, parent = #blocked6}>> to tensor<8xi64, #ttg.slice<{dim = 0, parent = #blocked6}>>
    %offset = tt.expand_dims %ext {axis = 0 : i32} : tensor<8xi64, #ttg.slice<{dim = 0, parent = #blocked6}>> -> tensor<1x8xi64, #blocked6>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>, #blocked6>
    %ptr = tt.addptr %base, %offset : tensor<1x8x!tt.ptr<f32>, #blocked6>, tensor<1x8xi64, #blocked6>
    tt.store %ptr, %arg1 : tensor<1x8x!tt.ptr<f32>, #blocked6>
    tt.store %ptr, %arg1 : tensor<1x8x!tt.ptr<f32>, #blocked6>
    // CHECK-COUNT-2: arith.trunci {{.*}} tensor<1x8xi64, {{.*}}> to tensor<1x8xi32, {{.*}}>
    // CHECK: amdg.buffer_store
    // CHECK: amdg.buffer_store
    // CHECK-NOT: tt.store
    tt.return
  }
}
