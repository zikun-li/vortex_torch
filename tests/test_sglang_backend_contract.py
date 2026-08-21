import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vortex_torch.indexer.context import Context


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


def test_attention_backends_do_not_read_removed_forward_batch_pool_attribute():
    backend_root = ROOT / "vortex_torch/engine/sgl/attention_backend"
    for filename in BACKENDS:
        assert "forward_batch.token_to_kv_pool" not in (backend_root / filename).read_text()


def _class_method(tree, class_name, method_name):
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _keyword_string(call, name):
    keyword = next(keyword for keyword in call.keywords if keyword.arg == name)
    assert isinstance(keyword.value, ast.Constant)
    return keyword.value.value


def test_trtllm_backend_compiles_distinct_decode_and_sparse_prefill_layouts():
    path = ROOT / "vortex_torch/engine/sgl/attention_backend/trtllm.py"
    tree = ast.parse(path.read_text())
    compile_method = _class_method(tree, "VortexTRTLLMBackend", "_compile")

    trace_calls = [
        node
        for node in ast.walk(compile_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_trace_and_compile"
    ]
    assert {
        _keyword_string(call, "attention_backend") for call in trace_calls
    } == {"trtllm", "flashinfer"}

    init = _class_method(tree, "VortexTRTLLMBackend", "__init__")
    assigned_calls = {
        target.attr: {
            call.func.id
            for call in ast.walk(node.value)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    assert assigned_calls["plan_decode"] == {"get_decode_planner_trtllm"}
    assert assigned_calls["plan_decode_prefill"] == {"get_decode_planner"}


def test_trtllm_sparse_prefill_reuses_the_verified_flashinfer_path():
    path = ROOT / "vortex_torch/engine/sgl/attention_backend/trtllm.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VortexTRTLLMBackend"
    )

    aliases = {
        target.id: ast.unparse(node.value)
        for node in cls.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance((target := node.targets[0]), ast.Name)
    }
    assert aliases["_forward_extend_sparse"] == (
        "VortexFlashInferBackend._forward_extend_sparse"
    )
    assert aliases["_apply_gt_prefill_patch"] == (
        "VortexFlashInferBackend._apply_gt_prefill_patch"
    )

    flash_path = ROOT / "vortex_torch/engine/sgl/attention_backend/flashinfer.py"
    flash_tree = ast.parse(flash_path.read_text())
    sparse_method = _class_method(
        flash_tree, "VortexFlashInferBackend", "_forward_extend_sparse"
    )
    planner_calls = [
        node
        for node in ast.walk(sparse_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"plan_decode", "plan_decode_prefill"}
    ]
    assert {node.func.attr for node in planner_calls} == {"plan_decode_prefill"}


def test_context_accepts_a_per_compile_attention_backend_override():
    server_args = SimpleNamespace(
        page_size=16,
        vortex_max_seq_lens=1000,
        max_prefill_tokens=128,
        vortex_workload_chunk_size=32,
        vortex_block_size=16,
        vortex_topk_val=29,
        vortex_max_topk_val=None,
        vortex_topk_ratio=0.0,
        vortex_dtype="bfloat16",
        vortex_block_reserved_bos=1,
        vortex_block_reserved_eos=2,
        vortex_compilation_cache_dir="/tmp",
        vortex_deterministic_topk=False,
        vortex_impl_backend="triton",
        vortex_attention_backend="trtllm",
        vortex_use_tensor_core=False,
    )
    model_runner = SimpleNamespace(
        server_args=server_args,
        model_config=SimpleNamespace(context_len=1000),
        req_to_token_pool=SimpleNamespace(size=4),
    )
    parent = SimpleNamespace(
        group_size=8,
        num_kv_heads=2,
        num_qo_heads=16,
        head_dim=128,
        sparse_attention=object(),
    )

    device = SimpleNamespace(multi_processor_count=132)
    with patch("torch.cuda.get_device_properties", return_value=device):
        decode_ctx = Context().create(parent, model_runner)
        prefill_ctx = Context().create(
            parent,
            model_runner,
            prefill=True,
            attention_backend="flashinfer",
        )

    assert decode_ctx.vortex_attention_backend == "trtllm"
    assert decode_ctx.max_num_blocks_per_request == 64
    assert prefill_ctx.vortex_attention_backend == "flashinfer"
    assert prefill_ctx.max_num_blocks_per_request == 63
    assert prefill_ctx.sparse_prefill is True
