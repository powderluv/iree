// RUN: iree-translate %s --iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=dylib-llvm-aot --iree-codegen-use-sandbox-passes -linalg-transform-file-name="%p/linalg_transform_spec.mlir"
// iree-check-module /tmp/aaa.vmfb --driver=dylib --entry_function=f32

func private @_f32() {
  %cst = arith.constant dense<[[4.300000e+02, 3.880000e+02, 3.460000e+02, 3.040000e+02, 2.620000e+02], [3.400000e+02, 3.070000e+02, 2.740000e+02, 2.410000e+02, 2.080000e+02], [2.500000e+02, 2.260000e+02, 2.020000e+02, 1.780000e+02, 1.540000e+02], [1.600000e+02, 1.450000e+02, 1.300000e+02, 1.150000e+02, 1.000000e+02], [7.000000e+01, 6.400000e+01, 5.800000e+01, 5.200000e+01, 4.600000e+01]]> : tensor<5x5xf32>
  %cst_0 = arith.constant dense<[[1.500000e+01, 1.400000e+01, 1.300000e+01, 1.200000e+01, 1.100000e+01], [1.000000e+01, 9.000000e+00, 8.000000e+00, 7.000000e+00, 6.000000e+00], [5.000000e+00, 4.000000e+00, 3.000000e+00, 2.000000e+00, 1.000000e+00]]> : tensor<3x5xf32>
  %cst_1 = arith.constant dense<[[1.500000e+01, 1.400000e+01, 1.300000e+01], [1.200000e+01, 1.100000e+01, 1.000000e+01], [9.000000e+00, 8.000000e+00, 7.000000e+00], [6.000000e+00, 5.000000e+00, 4.000000e+00], [3.000000e+00, 2.000000e+00, 1.000000e+00]]> : tensor<5x3xf32>
  %c5 = arith.constant 5 : index
  %c1 = arith.constant 1 : index
  %0 = util.do_not_optimize(%cst_1) : tensor<5x3xf32>
  %1 = util.do_not_optimize(%cst_0) : tensor<3x5xf32>
  %2 = flow.dispatch.workgroups[%c5, %c5, %c1](%0, %1) : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32> =
      (%arg0: !flow.dispatch.tensor<readonly:5x3xf32>, %arg1: !flow.dispatch.tensor<readonly:3x5xf32>, %arg2: !flow.dispatch.tensor<writeonly:5x5xf32>) {
    %cst_2 = arith.constant 0.000000e+00 : f32
    %3 = flow.dispatch.tensor.load %arg0, offsets = [0, 0], sizes = [5, 3], strides = [1, 1] : !flow.dispatch.tensor<readonly:5x3xf32> -> tensor<5x3xf32>
    %4 = flow.dispatch.tensor.load %arg1, offsets = [0, 0], sizes = [3, 5], strides = [1, 1] : !flow.dispatch.tensor<readonly:3x5xf32> -> tensor<3x5xf32>

    %5 = linalg.init_tensor [5, 5] : tensor<5x5xf32>
    %6 = linalg.fill ins(%cst_2 : f32) outs(%5: tensor<5x5xf32> ) -> tensor<5x5xf32> 
    %7 = linalg.matmul ins(%3, %4 : tensor<5x3xf32>, tensor<3x5xf32>) outs(%6 : tensor<5x5xf32>) -> tensor<5x5xf32>
    
    flow.dispatch.tensor.store %7, %arg2, offsets = [0, 0], sizes = [5, 5], strides = [1, 1] : tensor<5x5xf32> -> !flow.dispatch.tensor<writeonly:5x5xf32>

    flow.return
  }
  check.expect_almost_eq(%2, %cst) : tensor<5x5xf32>
  return
}
