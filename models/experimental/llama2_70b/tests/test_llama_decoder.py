# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from torch import nn
import tt_lib
import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor

from models.experimental.llama2_70b.reference.llama.llama import Llama
from models.experimental.llama2_70b.tt.llama_decoder_optimized import TtLlamaDecoder_optimized
from models.experimental.llama2_70b.reference.llama.llama.model import precompute_freqs_cis
from models.utility_functions import skip_for_grayskull
from models.experimental.llama2_70b.tt.llama_common import (
    setup_llama_env,
    check_device_mesh,
    extract_pcc_from_log,
    generate_rot_emb,
    get_rotation_mat,
    gather_cos_sin,
    precompute_freqs,
    MAX_SEQ_LEN,
    MAX_SEQ_LEN_LLAMA3,
    BASE_URL,
    UNIT_TEST_N_LAYER,
    UNIT_TEST_LAYER_NUM,
    UNIT_TEST_START_POS,
    UNIT_TEST_GENERATION_LENGTH,
    comp_pcc,
    get_rot_transformation_mat,
    should_skip_model_load,
    check_kv_cache,
)
from models.utility_functions import nearest_32


class PytorchLlamaDecoderModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num, rope_theta):
        super().__init__()
        self.decoder = hf_reference_model.layers[layer_num]
        self.rope_theta = rope_theta

        # Disable dropout
        self.decoder.eval()

        configuration = hf_reference_model.params
        self.n_heads = configuration.n_heads
        hidden_dim = configuration.dim
        self.head_dim = hidden_dim // self.n_heads
        self.max_seq_len = configuration.max_seq_len

    def prepare_inputs(self, x, start_pos):
        """
        Prepare inputs for decode mode. Assume that current token is at
        start_pos, and KV cache has valid data up to start_pos.
        """
        batch = x.size(0)
        freqs_cis = precompute_freqs_cis(self.head_dim, self.max_seq_len * 2, self.rope_theta)
        freqs_cis = freqs_cis[start_pos : start_pos + 1]

        attn_mask = torch.zeros(batch, 1, 1, start_pos + 1)
        # attn_mask[:, :, :, : start_pos + 1] = -1e9
        attn_mask = attn_mask.expand(-1, self.n_heads, -1, -1)

        return x, start_pos, freqs_cis, attn_mask

    def prepare_inputs_prefill(self, x, start_pos):
        """
        Prepare inputs for decode mode. Assume that current token is at
        start_pos, and KV cache has valid data up to start_pos.
        """
        batch = x.size(0)
        seq_len = x.size(1)
        freqs_cis = precompute_freqs_cis(self.head_dim, self.max_seq_len * 2, self.rope_theta)
        freqs_cis = freqs_cis[start_pos : start_pos + seq_len]

        attn_mask = torch.full((seq_len, seq_len), float("-inf"))
        attn_mask = torch.triu(attn_mask, diagonal=1)
        attn_mask = attn_mask.expand(batch, self.n_heads, -1, -1)

        return x, start_pos, freqs_cis, attn_mask

    def forward(self, x, start_pos, freqs_cis, mask):
        """
        x: (batch, seq, hidden_dim)
        start_pos: int
        freqs_cis: ?
        mask: ?

        return: (batch, seq, hidden_dim)
        """
        result = self.decoder(
            x,
            start_pos,
            freqs_cis,
            mask,
        )
        return result


def tt_llama_decoder_prepare_inputs(llama_decoder_model, x, start_pos):
    assert len(x.size()) == 3
    batch, seq_len, hidden_size = x.shape

    cache_name = lambda name: llama_decoder_model.cache_path / (
        f"{'llama3_' if llama_decoder_model.llama3 else ''}{name}"
    )

    if llama_decoder_model.model_config["LLM_MODE"] == "prefill":
        assert seq_len % 128 == 0 and seq_len > 0, "Prefill mode only supports seqlen as a multiple of 128 up to 2k"
        assert batch == 1, "prefill mode only supports batch size 1"
        x = x.unsqueeze(1)  # [batch, 1, seq_len, hidden_dim]

        xs = ttnn.as_tensor(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=llama_decoder_model.model_config["DRAM_MEMCFG"],
            mesh_mapper=ttnn.ShardTensorToMesh(llama_decoder_model.device_mesh, dim=3),
            device=llama_decoder_model.device_mesh,
        )
        xs = ttnn.to_device(xs, llama_decoder_model.device_mesh)

        cos, sin = precompute_freqs(
            llama_decoder_model.head_dim, llama_decoder_model.max_seq_len * 2, llama_decoder_model.rope_theta
        )
        cos_gathered, sin_gathered = gather_cos_sin(torch.arange(start_pos, start_pos + seq_len), cos, sin)
        assert cos_gathered.size() == (1, 1, seq_len, llama_decoder_model.head_dim)
        assert sin_gathered.size() == (1, 1, seq_len, llama_decoder_model.head_dim)

        cos_gathereds = ttnn.as_tensor(
            cos_gathered,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name(f"cos_gathered_prefill_{seq_len}"),
            memory_config=llama_decoder_model.model_config["DRAM_MEMCFG"],
            device=llama_decoder_model.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(llama_decoder_model.device_mesh),
        )
        sin_gathereds = ttnn.as_tensor(
            sin_gathered,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name(f"sin_gathered_prefill_{seq_len}"),
            memory_config=llama_decoder_model.model_config["DRAM_MEMCFG"],
            device=llama_decoder_model.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(llama_decoder_model.device_mesh),
        )
        cos_gathereds = ttnn.to_device(cos_gathereds, llama_decoder_model.device_mesh)
        sin_gathereds = ttnn.to_device(sin_gathereds, llama_decoder_model.device_mesh)
        rot_mats = [cos_gathereds, sin_gathereds]

        attn_mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        attn_mask = attn_mask.expand(batch, 1, -1, -1)
        attn_masks = ttnn.as_tensor(
            attn_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name(f"attn_mask_prefill_{seq_len}"),
            mesh_mapper=ReplicateTensorToMesh(llama_decoder_model.device_mesh),
            memory_config=llama_decoder_model.model_config["DRAM_MEMCFG"],
            device=llama_decoder_model.device_mesh,
        )
        attn_masks = ttnn.to_device(attn_masks, llama_decoder_model.device_mesh)

    elif llama_decoder_model.model_config["LLM_MODE"] == "decode":
        assert seq_len == 1, "Only supporting decode mode"
        x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, hidden_dim]

        xs = ttnn.as_tensor(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=llama_decoder_model.model_config["DRAM_MEMCFG"],
            mesh_mapper=ttnn.ShardTensorToMesh(llama_decoder_model.device_mesh, dim=3),
            device=llama_decoder_model.device_mesh,
        )
        xs = ttnn.to_device(xs, llama_decoder_model.device_mesh)
        xs = tt_lib.tensor.interleaved_to_sharded(
            xs, sharded_mem_config=llama_decoder_model.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"]
        )

        rot_emb = generate_rot_emb(
            llama_decoder_model.head_dim, llama_decoder_model.max_seq_len * 2, llama_decoder_model.rope_theta
        )
        rot_mat = get_rotation_mat(rot_emb, start_pos, seq_len, batch=batch)
        assert rot_mat.size() == (1, batch, llama_decoder_model.head_dim, llama_decoder_model.head_dim)
        rot_mats = ttnn.as_tensor(
            rot_mat,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=llama_decoder_model.model_config["DRAM_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(llama_decoder_model.device_mesh),
            device=llama_decoder_model.device_mesh,
        )
        rot_mats = ttnn.to_device(rot_mats, llama_decoder_model.device_mesh)

        rot_mats = tt_lib.tensor.interleaved_to_sharded(
            rot_mats, sharded_mem_config=llama_decoder_model.model_config["ROT_MAT_MM_IN1_MEMCFG"]
        )

        padded_layer_past_len = nearest_32(start_pos + 1)
        attn_mask_shape = (seq_len, 1, llama_decoder_model.padded_local_heads, padded_layer_past_len)
        attn_mask = torch.zeros(*attn_mask_shape)
        attn_mask[:, :, :, start_pos + 1 :] = torch.finfo(attn_mask.dtype).min

        attn_masks = ttnn.as_tensor(
            attn_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=llama_decoder_model.model_config["DRAM_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(llama_decoder_model.device_mesh),
            device=llama_decoder_model.device_mesh,
        )
        attn_masks = ttnn.to_device(attn_masks, llama_decoder_model.device_mesh)

        repeat_shape = (1, batch, 1, 1)
        attn_masks = tt_lib.tensor.repeat(
            attn_masks, repeat_shape, output_mem_config=llama_decoder_model.model_config["DRAM_MEMCFG"]
        )
        # Put attn_mask on the device with the sharded config
        attention_mask_memconfig = llama_decoder_model.model_config["ATTN_MASK_MEMCFG"]
        if attention_mask_memconfig.is_sharded():
            attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
            attn_mask_shard_shape[-1] = padded_layer_past_len
            attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape

            attn_masks = tt_lib.tensor.interleaved_to_sharded(attn_masks, sharded_mem_config=attention_mask_memconfig)

    return (
        xs,
        start_pos,
        rot_mats,
        attn_masks,
    )


def run_test_LlamaDecoder_inference(
    t3k_device_mesh,
    batch,
    seq_len,
    pcc,
    model_config,
    llama_version,
    ckpt_dir,
    tokenizer_path,
    cache_path,
):
    # Prepare paths and devices
    skip_model_load = should_skip_model_load()

    # Prepare configs
    hugging_face_reference_model = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len=MAX_SEQ_LEN if llama_version == "llama2" else MAX_SEQ_LEN_LLAMA3,
        max_batch_size=batch,
        n_layers=UNIT_TEST_N_LAYER,
        skip_model_load=skip_model_load,
    ).model
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()
    logger.info(state_dict.keys())
    torch.manual_seed(0)
    configuration = hugging_face_reference_model.params

    # PyTorch model --------------------------------------------------------------------
    pytorch_LlamaDecoder_model = PytorchLlamaDecoderModel(
        hugging_face_reference_model, UNIT_TEST_LAYER_NUM, configuration.rope_theta
    )
    # TT model -------------------------------------------------------------------------
    transformation_mat_torch = get_rot_transformation_mat(32)  # 32 for tile size
    transformation_mats = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_device_mesh,
        memory_config=model_config["DRAM_MEMCFG"],
        mesh_mapper=ReplicateTensorToMesh(t3k_device_mesh),
    )
    transformation_mats = ttnn.to_device(transformation_mats, t3k_device_mesh)

    tt_LlamaDecoder_model = TtLlamaDecoder_optimized(
        t3k_device_mesh,
        state_dict,
        BASE_URL,
        UNIT_TEST_LAYER_NUM,
        model_config,
        configuration,
        batch,
        transformation_mats,
        cache_path=cache_path,
    )

    all_tests_pass, all_pccs = True, []
    if model_config["LLM_MODE"] == "prefill":
        generation_start_pos = 0
        generation_length = 1
    else:
        generation_start_pos = UNIT_TEST_START_POS
        generation_length = UNIT_TEST_GENERATION_LENGTH
    for i in range(generation_length):
        # Prepare input
        pt_inp_ids = torch.randint(0, configuration.vocab_size, (batch, seq_len))
        pt_inp = hugging_face_reference_model.tok_embeddings(pt_inp_ids)
        tt_input = pt_inp.clone()
        start_pos = generation_start_pos + i

        # PyTorch output --------------------------------------------------------------------
        if model_config["LLM_MODE"] == "prefill":
            x_input, start_pos, freqs_cis, attn_mask = pytorch_LlamaDecoder_model.prepare_inputs_prefill(
                pt_inp, start_pos
            )
        else:
            x_input, start_pos, freqs_cis, attn_mask = pytorch_LlamaDecoder_model.prepare_inputs(pt_inp, start_pos)

        pytorch_out = pytorch_LlamaDecoder_model(
            x_input,
            start_pos,
            freqs_cis,
            attn_mask,
        )

        # TT hardware execution -------------------------------------------------------------
        x_input, start_pos, rot_mat, attn_mask = tt_llama_decoder_prepare_inputs(
            tt_LlamaDecoder_model, tt_input, start_pos
        )

        tt_out = tt_LlamaDecoder_model(
            x_input,
            rot_mat,
            start_pos,
            attn_mask,
        )

        tt_out = ttnn.from_device(tt_out)
        tt_out = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=3))
        tt_out = tt_out.permute(2, 1, 0, 3).squeeze(1)  # [seq, batch, hidden_dim]

        # check outputs ----------------------------------------------------------------------
        does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
        logger.info(f"Output: {output_pcc}")
        all_pccs.append(extract_pcc_from_log(output_pcc))

        if does_pass:
            logger.info(f"[start_pos={start_pos}] {llama_version} Decoder output Passed!")
        else:
            logger.warning(
                f"[start_pos={start_pos}] {llama_version} Decoder output Failed! PCC value is lower than {pcc}"
            )
            all_tests_pass = False

    logger.info(f"Average PCC over {len(all_pccs)} tokens: {sum(all_pccs) / len(all_pccs)}")
    # Check kv cache
    # PyTorch output --------------------------------------------------------------------
    pytorch_layer_present = [
        pytorch_LlamaDecoder_model.decoder.attention.cache_k.clone().permute(
            0, 2, 1, 3
        ),  # [batch, n_kv_heads, seq, head_dim]
        pytorch_LlamaDecoder_model.decoder.attention.cache_v.clone().permute(
            0, 2, 1, 3
        ),  # [batch, n_kv_heads, seq, head_dim]
    ]
    # TT hardware output -----------------------------------------------------------------

    tt_layer_present_all = [ttnn.from_device(lp) for lp in tt_LlamaDecoder_model.attention.layer_past]
    tt_layer_present_all = [
        ttnn.to_torch(lp, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=0)).transpose(0, 1)
        for lp in tt_layer_present_all
    ]

    cache_test_pass = check_kv_cache(
        pytorch_layer_present,
        tt_layer_present_all,
        generation_start_pos,
        generation_length,
        seq_len,
        model_config["LLM_MODE"] == "prefill",
        pcc,
    )
    all_tests_pass = all_tests_pass and cache_test_pass

    if all_tests_pass:
        logger.info(f"{llama_version} Decoder output Passed!")
    else:
        logger.warning(f"{llama_version} Decoder output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "llama_version",
    (
        ("llama2"),
        ("llama3"),
    ),
)
@pytest.mark.parametrize(
    "batch, seq_len, pcc",
    ((32, 1, 0.9993), (1, 128, 0.998), (1, 2048, 0.998), (1, 8192, 0.998)),
    ids=("decode", "prefill_128", "prefill_2k", "prefill_8k"),
)
def test_LlamaDecoder_inference(
    batch,
    seq_len,
    pcc,
    t3k_device_mesh,
    llama_version,
    use_program_cache,
):
    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version, batch=batch, seq_len=seq_len
    )

    check_device_mesh(t3k_device_mesh, model_config)
    run_test_LlamaDecoder_inference(
        t3k_device_mesh,
        batch,
        seq_len,
        pcc,
        model_config,
        llama_version,
        ckpt_dir,
        tokenizer_path,
        cache_path,
    )
