#include <torch/extension.h>

#include "pybind_common.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_num_blocks_per_seq", &get_num_blocks_per_seq,
          py::arg("M"), py::arg("E"));
    m.def("get_fc1_act_sf_size", &get_fc1_act_sf_size,
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("hidden_size"));
    m.def("get_fc2_act_sf_size", &get_fc2_act_sf_size,
          py::arg("M"), py::arg("E"), py::arg("topk"),
          py::arg("inter_size"));

    register_routing(m);
    register_expand_input_rows(m);
    register_up_gate(m);
    register_down(m);
    register_e512_topk10(m);
}
