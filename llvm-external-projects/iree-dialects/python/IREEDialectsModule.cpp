// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects-c/Dialects.h"
#include "iree-dialects-c/Utils.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/Diagnostics.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_ireeDialects, m) {
  m.doc() = "iree-dialects main python extension";

  auto irModule = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"));
  auto typeClass = irModule.attr("Type");

  //===--------------------------------------------------------------------===//
  // Utils
  //===--------------------------------------------------------------------===//

  m.def(
      "lookup_nearest_symbol_from",
      [](MlirOperation fromOp, MlirAttribute symbol) {
        if (!mlirAttributeIsASymbolRef(symbol)) {
          throw std::invalid_argument("expected a SymbolRefAttr");
        }
        return ireeLookupNearestSymbolFrom(fromOp, symbol);
      },
      py::arg("fromOp"), py::arg("symbol"));

  // TODO: Upstream this into the main Python bindings.
  m.def(
      "emit_error",
      [](MlirLocation loc, std::string message) {
        mlirEmitError(loc, message.c_str());
      },
      py::arg("loc"), py::arg("message"));

  //===--------------------------------------------------------------------===//
  // IREEDialect
  //===--------------------------------------------------------------------===//
  auto iree_m = m.def_submodule("iree_input");
  iree_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__iree_input__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  //===--------------------------------------------------------------------===//
  // IREELinalgExt
  //===--------------------------------------------------------------------===//
  auto iree_linalg_ext_m = m.def_submodule("iree_linalg_ext");
  iree_linalg_ext_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__iree_linalg_ext__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  //===--------------------------------------------------------------------===//
  // LinalgTransform
  //===--------------------------------------------------------------------===//
  auto iree_linalg_transform_m = m.def_submodule("iree_linalg_transform");
  mlirIREELinalgTransformRegisterPasses();
  iree_linalg_transform_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle =
            mlirGetDialectHandle__iree_linalg_transform__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  //===--------------------------------------------------------------------===//
  // TransformDialect
  //===--------------------------------------------------------------------===//
  auto transform_m = m.def_submodule("transform");
  mlirIREETransformRegisterPasses();

  transform_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__transform__();
        mlirDialectHandleRegisterDialect(handle, context);
        ireeRegisterTransformExtensions(context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
