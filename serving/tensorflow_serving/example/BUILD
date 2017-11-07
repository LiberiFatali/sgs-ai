# Description: Tensorflow Serving examples.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = [
        "-parse_headers",
        "no_layering_check",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

py_library(
    name = "mnist_input_data",
    srcs = ["mnist_input_data.py"],
)

py_binary(
    name = "mnist_export",
    srcs = [
        "mnist_export.py",
    ],
    deps = [
        ":mnist_input_data",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/contrib/session_bundle:exporter",
    ],
)

py_binary(
    name = "mnist_client",
    srcs = [
        "mnist_client.py",
    ],
    deps = [
        ":mnist_input_data",
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "inception_export",
    srcs = [
        "inception_export.py",
    ],
    deps = [
        "@inception_model//inception",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/contrib/session_bundle:exporter",
    ],
)

py_binary(
    name = "inception_client",
    srcs = [
        "inception_client.py",
    ],
    deps = [
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

# HieuNM
serving_proto_library(
    name = "inception_inference_proto",
    srcs = ["inception_inference.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
)

cc_binary(
    name = "sgsnet_1017_inception_inference",
    srcs = [
        "sgsnet_1017_inception_inference.cc",
    ],
    linkopts = ["-lm"],
    deps = [
        ":inception_inference_proto",
        "//tensorflow_serving/batching:basic_batch_scheduler",
        "//tensorflow_serving/batching:batch_scheduler",
        "//tensorflow_serving/core:manager",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core:servable_id",
        "//tensorflow_serving/servables/tensorflow:simple_servers",
        "@grpc//:grpc++",
        "@org_tensorflow//tensorflow/contrib/session_bundle",
        "@org_tensorflow//tensorflow/contrib/session_bundle:manifest_proto_cc",
        "@org_tensorflow//tensorflow/contrib/session_bundle:signature",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

py_binary(
    name = "sgsnet_1017_oversample_inception_export",
    srcs = [
        "sgsnet_1017_oversample_inception_export.py",
    ],
    deps = [
        "@inception_model//inception",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/contrib/session_bundle:exporter",
    ],
)

py_binary(
    name = "sgsnet_inception_client",
    srcs = [
        "sgsnet_inception_client.py",
        "inception_inference_pb2.py",
    ],
    deps = ["@org_tensorflow//tensorflow:tensorflow_py"],
)

py_binary(
    name = "sgsnet_1017_ir_v2_export_oversample",
    srcs = [
        "sgsnet_1017_ir_v2_export_oversample.py",
    ],
    deps = [
        "@inception_model//inception",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/contrib/session_bundle:exporter",
    ],
)
