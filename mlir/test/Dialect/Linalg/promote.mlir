// RUN: mlir-opt %s -linalg-promote-subviews | FileCheck %s
// RUN: mlir-opt %s -linalg-promote-subviews="test-promote-dynamic" | FileCheck %s --check-prefix=DYNAMIC

#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 4)>
#map3 = affine_map<(d0) -> (d0 + 3)>

// CHECK-DAG: #[[strided2D:.*]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
// CHECK-DAG: #[[strided2D_dynamic:.*]] = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>

func @matmul_f32(%A: memref<?xi8>, %M: index, %N: index, %K: index) {
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %3 = view %A[%c0][%M, %K] : memref<?xi8> to memref<?x?xf32>
  %4 = view %A[%c0][%K, %N] : memref<?xi8> to memref<?x?xf32>
  %5 = view %A[%c0][%M, %N] : memref<?xi8> to memref<?x?xf32>
  %6 = dim %3, 0 : memref<?x?xf32>
  %7 = dim %3, 1 : memref<?x?xf32>
  %8 = dim %4, 1 : memref<?x?xf32>
  scf.for %arg4 = %c0 to %6 step %c2 {
    scf.for %arg5 = %c0 to %8 step %c3 {
      scf.for %arg6 = %c0 to %7 step %c4 {
        %11 = std.subview %3[%arg4, %arg6][%c2, %c4][1, 1] : memref<?x?xf32> to memref<?x?xf32, offset: ?, strides: [?, 1]>
        %14 = std.subview %4[%arg6, %arg5][%c4, %c3][1, 1] : memref<?x?xf32> to memref<?x?xf32, offset: ?, strides: [?, 1]>
        %17 = std.subview %5[%arg4, %arg5][%c2, %c3][1, 1] : memref<?x?xf32> to memref<?x?xf32, offset: ?, strides: [?, 1]>
        linalg.matmul(%11, %14, %17) : memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>, memref<?x?xf32, offset: ?, strides: [?, 1]>
      }
    }
  }
  return
}

// CHECK-LABEL: func @matmul_f32(%{{.*}}: memref<?xi8>, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:         %[[vA:.*]] = subview {{.*}} : memref<?x?xf32>
//       CHECK:         %[[vB:.*]] = subview {{.*}} : memref<?x?xf32>
//       CHECK:         %[[vC:.*]] = subview {{.*}} : memref<?x?xf32>
///
//       CHECK:         %[[tmpA:.*]] = alloc() : memref<32xi8>
//       CHECK:         %[[fullA:.*]] = std.view %[[tmpA]][{{.*}}][{{.*}}] : memref<32xi8> to memref<?x?xf32>
//     DYNAMIC:         std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECK:         %[[partialA:.*]] = subview %[[fullA]]{{.*}} : memref<?x?xf32> to memref<?x?xf32, #[[strided2D_dynamic]]>
///
//       CHECK:         %[[tmpB:.*]] = alloc() : memref<48xi8>
//       CHECK:         %[[fullB:.*]] = std.view %[[tmpB]][{{.*}}][{{.*}}] : memref<48xi8> to memref<?x?xf32>
//     DYNAMIC:         std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECK:         %[[partialB:.*]] = subview %[[fullB]]{{.*}} : memref<?x?xf32> to memref<?x?xf32, #[[strided2D_dynamic]]>
///
//       CHECK:         %[[tmpC:.*]] = alloc() : memref<24xi8>
//       CHECK:         %[[fullC:.*]] = std.view %[[tmpC]][{{.*}}][{{.*}}] : memref<24xi8> to memref<?x?xf32>
//     DYNAMIC:         std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?x?xf32>
//       CHECK:         %[[partialC:.*]] = subview %[[fullC]]{{.*}} : memref<?x?xf32> to memref<?x?xf32, #[[strided2D_dynamic]]>

//       CHECK:         linalg.fill(%[[fullA]], {{.*}}) : memref<?x?xf32>, f32
//       CHECK:         linalg.fill(%[[fullB]], {{.*}}) : memref<?x?xf32>, f32
//       CHECK:         linalg.fill(%[[fullC]], {{.*}}) : memref<?x?xf32>, f32
//       CHECK:         linalg.copy(%[[vA]], %[[partialA]]) : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D_dynamic]]>
//       CHECK:         linalg.copy(%[[vB]], %[[partialB]]) : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D_dynamic]]>
//       CHECK:         linalg.copy(%[[vC]], %[[partialC]]) : memref<?x?xf32, #[[strided2D]]>, memref<?x?xf32, #[[strided2D_dynamic]]>
//
//       CHECK:         linalg.matmul(%[[fullA]], %[[fullB]], %[[fullC]]) : memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
//
//       CHECK:         linalg.copy(%[[partialC]], %[[vC]]) : memref<?x?xf32, #[[strided2D_dynamic]]>, memref<?x?xf32, #[[strided2D]]>
//
//       CHECK:         dealloc %[[tmpA]] : memref<32xi8>
//       CHECK:         dealloc %[[tmpB]] : memref<48xi8>
//       CHECK:         dealloc %[[tmpC]] : memref<24xi8>

// -----

func @matmul_f64(%A: memref<?xi8>, %M: index, %N: index, %K: index) {
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %3 = view %A[%c0][%M, %K] : memref<?xi8> to memref<?x?xf64>
  %4 = view %A[%c0][%K, %N] : memref<?xi8> to memref<?x?xf64>
  %5 = view %A[%c0][%M, %N] : memref<?xi8> to memref<?x?xf64>
  %6 = dim %3, 0 : memref<?x?xf64>
  %7 = dim %3, 1 : memref<?x?xf64>
  %8 = dim %4, 1 : memref<?x?xf64>
  scf.for %arg4 = %c0 to %6 step %c2 {
    scf.for %arg5 = %c0 to %8 step %c3 {
      scf.for %arg6 = %c0 to %7 step %c4 {
        %11 = std.subview %3[%arg4, %arg6][%c2, %c4][1, 1] : memref<?x?xf64> to memref<?x?xf64, offset: ?, strides: [?, 1]>
        %14 = std.subview %4[%arg6, %arg5][%c4, %c3][1, 1] : memref<?x?xf64> to memref<?x?xf64, offset: ?, strides: [?, 1]>
        %17 = std.subview %5[%arg4, %arg5][%c2, %c3][1, 1] : memref<?x?xf64> to memref<?x?xf64, offset: ?, strides: [?, 1]>
        linalg.matmul(%11, %14, %17) : memref<?x?xf64, offset: ?, strides: [?, 1]>, memref<?x?xf64, offset: ?, strides: [?, 1]>, memref<?x?xf64, offset: ?, strides: [?, 1]>
      }
    }
  }
  return
}

// CHECK-LABEL: func @matmul_f64(%{{.*}}: memref<?xi8>, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:         %[[vA_f64:.*]] = subview {{.*}} : memref<?x?xf64>
//       CHECK:         %[[vB_f64:.*]] = subview {{.*}} : memref<?x?xf64>
//       CHECK:         %[[vC_f64:.*]] = subview {{.*}} : memref<?x?xf64>
///
//       CHECK:         %[[tmpA_f64:.*]] = alloc() : memref<64xi8>
//       CHECK:         %[[fullA_f64:.*]] = std.view %[[tmpA_f64]][{{.*}}][{{.*}}] : memref<64xi8> to memref<?x?xf64>
//     DYNAMIC:         std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?x?xf64>
//       CHECK:         %[[partialA_f64:.*]] = subview %[[fullA_f64]][%{{.*}}, %{{.*}}] : memref<?x?xf64> to memref<?x?xf64, #[[strided2D_dynamic]]>
///
//       CHECK:         %[[tmpB_f64:.*]] = alloc() : memref<96xi8>
//       CHECK:         %[[fullB_f64:.*]] = std.view %[[tmpB_f64]][{{.*}}][{{.*}}] : memref<96xi8> to memref<?x?xf64>
//     DYNAMIC:         std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?x?xf64>
//       CHECK:         %[[partialB_f64:.*]] = subview %[[fullB_f64]][%{{.*}}, %{{.*}}] : memref<?x?xf64> to memref<?x?xf64, #[[strided2D_dynamic]]>
///
//       CHECK:         %[[tmpC_f64:.*]] = alloc() : memref<48xi8>
//       CHECK:         %[[fullC_f64:.*]] = std.view %[[tmpC_f64]][{{.*}}][{{.*}}] : memref<48xi8> to memref<?x?xf64>
//     DYNAMIC:         std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?x?xf64>
//       CHECK:         %[[partialC_f64:.*]] = subview %[[fullC_f64]][%{{.*}}, %{{.*}}] : memref<?x?xf64> to memref<?x?xf64, #[[strided2D_dynamic]]>

//       CHECK:         linalg.fill(%[[fullA_f64]], {{.*}}) : memref<?x?xf64>, f64
//       CHECK:         linalg.fill(%[[fullB_f64]], {{.*}}) : memref<?x?xf64>, f64
//       CHECK:         linalg.fill(%[[fullC_f64]], {{.*}}) : memref<?x?xf64>, f64
//       CHECK:         linalg.copy(%[[vA_f64]], %[[partialA_f64]]) : memref<?x?xf64, #[[strided2D]]>, memref<?x?xf64, #[[strided2D_dynamic]]>
//       CHECK:         linalg.copy(%[[vB_f64]], %[[partialB_f64]]) : memref<?x?xf64, #[[strided2D]]>, memref<?x?xf64, #[[strided2D_dynamic]]>
//       CHECK:         linalg.copy(%[[vC_f64]], %[[partialC_f64]]) : memref<?x?xf64, #[[strided2D]]>, memref<?x?xf64, #[[strided2D_dynamic]]>
//
//       CHECK:         linalg.matmul(%[[fullA_f64]], %[[fullB_f64]], %[[fullC_f64]]) : memref<?x?xf64>, memref<?x?xf64>, memref<?x?xf64>
//
//       CHECK:         linalg.copy(%[[partialC_f64]], %[[vC_f64]]) : memref<?x?xf64, #[[strided2D_dynamic]]>, memref<?x?xf64, #[[strided2D]]>
//
//       CHECK:         dealloc %[[tmpA_f64]] : memref<64xi8>
//       CHECK:         dealloc %[[tmpB_f64]] : memref<96xi8>
//       CHECK:         dealloc %[[tmpC_f64]] : memref<48xi8>

// -----

func @matmul_i32(%A: memref<?xi8>, %M: index, %N: index, %K: index) {
  %c4 = constant 4 : index
  %c3 = constant 3 : index
  %c2 = constant 2 : index
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %3 = view %A[%c0][%M, %K] : memref<?xi8> to memref<?x?xi32>
  %4 = view %A[%c0][%K, %N] : memref<?xi8> to memref<?x?xi32>
  %5 = view %A[%c0][%M, %N] : memref<?xi8> to memref<?x?xi32>
  %6 = dim %3, 0 : memref<?x?xi32>
  %7 = dim %3, 1 : memref<?x?xi32>
  %8 = dim %4, 1 : memref<?x?xi32>
  scf.for %arg4 = %c0 to %6 step %c2 {
    scf.for %arg5 = %c0 to %8 step %c3 {
      scf.for %arg6 = %c0 to %7 step %c4 {
        %11 = std.subview %3[%arg4, %arg6][%c2, %c4][1, 1] : memref<?x?xi32> to memref<?x?xi32, offset: ?, strides: [?, 1]>
        %14 = std.subview %4[%arg6, %arg5][%c4, %c3][1, 1] : memref<?x?xi32> to memref<?x?xi32, offset: ?, strides: [?, 1]>
        %17 = std.subview %5[%arg4, %arg5][%c2, %c3][1, 1] : memref<?x?xi32> to memref<?x?xi32, offset: ?, strides: [?, 1]>
        linalg.matmul(%11, %14, %17) : memref<?x?xi32, offset: ?, strides: [?, 1]>, memref<?x?xi32, offset: ?, strides: [?, 1]>, memref<?x?xi32, offset: ?, strides: [?, 1]>
      }
    }
  }
  return
}

// CHECK-LABEL: func @matmul_i32(%{{.*}}: memref<?xi8>, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//       CHECK:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
//       CHECK:         %[[vA_i32:.*]] = subview {{.*}} : memref<?x?xi32>
//       CHECK:         %[[vB_i32:.*]] = subview {{.*}} : memref<?x?xi32>
//       CHECK:         %[[vC_i32:.*]] = subview {{.*}} : memref<?x?xi32>
///
//       CHECK:         %[[tmpA_i32:.*]] = alloc() : memref<32xi8>
//       CHECK:         %[[fullA_i32:.*]] = std.view %[[tmpA_i32]][{{.*}}][{{.*}}] : memref<32xi8> to memref<?x?xi32>
//     DYNAMIC:         std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?x?xi32>
//       CHECK:         %[[partialA_i32:.*]] = subview %[[fullA_i32]][%{{.*}}, %{{.*}}] : memref<?x?xi32> to memref<?x?xi32, #[[strided2D_dynamic]]>
///
//       CHECK:         %[[tmpB_i32:.*]] = alloc() : memref<48xi8>
//       CHECK:         %[[fullB_i32:.*]] = std.view %[[tmpB_i32]][{{.*}}][{{.*}}] : memref<48xi8> to memref<?x?xi32>
//     DYNAMIC:         std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?x?xi32>
//       CHECK:         %[[partialB_i32:.*]] = subview %[[fullB_i32]][%{{.*}}, %{{.*}}] : memref<?x?xi32> to memref<?x?xi32, #[[strided2D_dynamic]]>
///
//       CHECK:         %[[tmpC_i32:.*]] = alloc() : memref<24xi8>
//       CHECK:         %[[fullC_i32:.*]] = std.view %[[tmpC_i32]][{{.*}}][{{.*}}] : memref<24xi8> to memref<?x?xi32>
//     DYNAMIC:         std.view %{{.*}}[{{.*}}][{{.*}}] : memref<?xi8> to memref<?x?xi32>
//       CHECK:         %[[partialC_i32:.*]] = subview %[[fullC_i32]][%{{.*}}, %{{.*}}] : memref<?x?xi32> to memref<?x?xi32, #[[strided2D_dynamic]]>

//       CHECK:         linalg.fill(%[[fullA_i32]], {{.*}}) : memref<?x?xi32>, i32
//       CHECK:         linalg.fill(%[[fullB_i32]], {{.*}}) : memref<?x?xi32>, i32
//       CHECK:         linalg.fill(%[[fullC_i32]], {{.*}}) : memref<?x?xi32>, i32
//       CHECK:         linalg.copy(%[[vA_i32]], %[[partialA_i32]]) : memref<?x?xi32, #[[strided2D]]>, memref<?x?xi32, #[[strided2D_dynamic]]>
//       CHECK:         linalg.copy(%[[vB_i32]], %[[partialB_i32]]) : memref<?x?xi32, #[[strided2D]]>, memref<?x?xi32, #[[strided2D_dynamic]]>
//       CHECK:         linalg.copy(%[[vC_i32]], %[[partialC_i32]]) : memref<?x?xi32, #[[strided2D]]>, memref<?x?xi32, #[[strided2D_dynamic]]>
//
//       CHECK:         linalg.matmul(%[[fullA_i32]], %[[fullB_i32]], %[[fullC_i32]]) : memref<?x?xi32>, memref<?x?xi32>, memref<?x?xi32>
//
//       CHECK:         linalg.copy(%[[partialC_i32]], %[[vC_i32]]) : memref<?x?xi32, #[[strided2D_dynamic]]>, memref<?x?xi32, #[[strided2D]]>
//
//       CHECK:         dealloc %[[tmpA_i32]] : memref<32xi8>
//       CHECK:         dealloc %[[tmpB_i32]] : memref<48xi8>
//       CHECK:         dealloc %[[tmpC_i32]] : memref<24xi8>
