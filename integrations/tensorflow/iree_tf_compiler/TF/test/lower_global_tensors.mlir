// RUN: iree-tf-opt --iree-tf-saved-model-lower-global-tensors --split-input-file %s | FileCheck %s

// CHECK-LABEL: module attributes {tf_saved_model.semantics}
module attributes {tf_saved_model.semantics} {

// TODO(silvasean): Verify "type" handling.
// I think when "type" is a partial type that flow will not model it correctly.

// CHECK:       iree_input.global private mutable [[V:@.+]] : tensor<?xf32> = dense<1.000000e+00> : tensor<1xf32>
// CHECK:       func.func @f() -> (tensor<?xf32> {tf_saved_model.index_path = []})
// CHECK-NEXT:    [[PTR:%.+]] = iree_input.global.address [[V]] : !iree_input.ptr<tensor<?xf32>>
// CHECK-NEXT:    [[T:%.+]] = iree_input.global.load.indirect [[PTR]] : !iree_input.ptr<tensor<?xf32>> -> tensor<?xf32>
// CHECK-NEXT:    return [[T]] : tensor<?xf32>

  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  func.func @f(%arg0: tensor<!tf_type.resource<tensor<?xf32>>> {tf_saved_model.bound_input = @v})
  -> (tensor<?xf32> {tf_saved_model.index_path = []})
  attributes {tf_saved_model.exported_names = ["f"]} {
    %0 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<?xf32>>>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
}

// -----

// CHECK-LABEL: module attributes {tf_saved_model.semantics}
module attributes {tf_saved_model.semantics} {

// CHECK:       iree_input.global private mutable [[V:@.+]] : tensor<?xf32> = dense<1.000000e+00> : tensor<1xf32>
// CHECK:       func.func @f(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]})
// CHECK-NEXT:    [[PTR:%.+]] = iree_input.global.address [[V]] : !iree_input.ptr<tensor<?xf32>>
// CHECK-NEXT:    iree_input.global.store.indirect %arg0, [[PTR]] : tensor<?xf32> -> !iree_input.ptr<tensor<?xf32>>
// CHECK-NEXT:    return

  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  func.func @f(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]}, %arg1: tensor<!tf_type.resource<tensor<?xf32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    "tf.AssignVariableOp"(%arg1, %arg0) : (tensor<!tf_type.resource<tensor<?xf32>>>, tensor<?xf32>) -> ()
    return
  }
}
