import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKENDS = {
    "flashinfer.py": "VortexFlashInferBackend",
    "trtllm.py": "VortexTRTLLMBackend",
    "trtllm_mla.py": "VortexTRTLLMMLABackend",
    "triton_mla.py": "VortexTritonMLABackend",
    "cuda_mla.py": "VortexCudaMLABackend",
}


@pytest.mark.parametrize(("filename", "class_name"), BACKENDS.items())
def test_attention_backend_exposes_hybrid_dispatcher_pool_contract(filename, class_name):
    path = ROOT / "vortex_torch/engine/sgl/attention_backend" / filename
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    init = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assigned = {
        target.attr
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    assert {"token_to_kv_pool", "req_to_token_pool"} <= assigned
