"""Adapter selection (Renderer(adapter=...), --adapter): the matching, on made-up adapters."""

import pytest

from replica_wgpu.renderer import match_adapter

ADAPTERS = [
    {"device": "NVIDIA GeForce RTX 5090", "backend_type": "Vulkan", "adapter_type": "DiscreteGPU"},
    {"device": "AMD Ryzen 9 9950X 16-Core Processor (RADV RAPHAEL_MENDOCINO)", "backend_type": "Vulkan",
     "adapter_type": "IntegratedGPU"},
    {"device": "llvmpipe (LLVM 21.1.8, 256 bits)", "backend_type": "Vulkan", "adapter_type": "CPU"},
    {"device": "NVIDIA GeForce RTX 5090/PCIe/SSE2", "backend_type": "OpenGL", "adapter_type": "Unknown"},
]


@pytest.mark.parametrize("spec, index", [
    ("discrete-gpu", 0), ("DiscreteGPU", 0), ("integrated-gpu", 1), ("cpu", 2),  # adapter type
    ("5090", 0), ("nvidia", 0), ("radv", 1), ("LLVMpipe", 2), ("ryzen", 1),      # device name
    ("vulkan", 0), ("opengl", 3),                                              # backend
    ("gpu", 0),                                                                # part of a type
])
def test_match(spec, index):
    assert match_adapter(spec, ADAPTERS) == index


def test_vulkan_preferred_over_opengl():
    assert match_adapter("5090", ADAPTERS[::-1]) == 3  # the Vulkan one, though listed last


def test_cpu_is_the_type_not_a_device_name():
    infos = [{"device": "Some CPU-brand iGPU", "backend_type": "Vulkan", "adapter_type": "IntegratedGPU"},
             ADAPTERS[2]]
    assert match_adapter("cpu", infos) == 1


def test_no_match_lists_what_is_available():
    with pytest.raises(ValueError, match="(?s)no GPU adapter matches 'voodoo'.*RTX 5090.*llvmpipe"):
        match_adapter("voodoo", ADAPTERS)
