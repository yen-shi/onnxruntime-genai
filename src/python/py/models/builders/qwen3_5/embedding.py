# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import copy
import os

import onnx_ir as ir
import torch
from loaders.vision import load_visual_state

from ..base import Model


class Qwen35EmbeddingModel(Model):
    """Merge token embeddings and image features, including image-free decode steps."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.image_token_id = config.image_token_id
        options = {"filename": "embedding.onnx", "hf_token": extra_options.get("hf_token", True)}
        super().__init__(copy.deepcopy(config), io_dtype, io_dtype, ep, cache_dir, options)

    def load_weights(self, input_path):
        source = input_path if os.path.isdir(input_path) else self.model_name_or_path
        prefix = "model.language_model.embed_tokens."
        return {
            name.removeprefix(prefix): tensor
            for name, tensor in load_visual_state(
                source, prefixes=(prefix,), cache_dir=self.cache_dir, token=self.hf_token
            ).items()
        }

    def make_model(self, input_path):
        self.weights = self.load_weights(input_path)
        hidden_size = self.hidden_size
        self.model.graph.inputs.append(
            self.make_value("input_ids", ir.DataType.INT64, ["batch_size", "sequence_length"])
        )
        ids = "input_ids"
        self.model.graph.inputs.append(
            self.make_value("image_features", self.io_dtype, ["num_image_tokens", hidden_size])
        )
        features = "image_features"
        zero = "/model/constants/INT64/0"
        one = "/model/constants/INT64/1"
        image_token = f"/model/constants/INT64/{self.image_token_id}"
        self.make_equal("embedding/image_mask", [ids, image_token], shape=None)
        mask = "embedding/image_mask/output_0"
        self.make_where("embedding/safe_ids", [mask, zero, ids], dtype=ir.DataType.INT64, shape=None)
        safe_ids = "embedding/safe_ids/output_0"
        self.make_initializer(self.weights["weight"], "embed_tokens.weight", to=self.io_dtype)
        self.make_gather("embedding/Gather", ["embed_tokens.weight", safe_ids], dtype=self.io_dtype, shape=None, axis=0)
        embedded = "embedding/Gather/output_0"
        flat_shape = "/model/constants/INT64/[-1]"
        self.make_reshape("embedding/flat_mask", [mask, flat_shape], dtype=ir.DataType.BOOL, shape=None)
        flat_mask = "embedding/flat_mask/output_0"
        self.make_cast("embedding/Cast", flat_mask, dtype=ir.DataType.INT64, shape=None)
        flat_mask = "embedding/Cast/output_0"
        self.make_cumsum("embedding/CumSum", [flat_mask, zero], dtype=ir.DataType.INT64, shape=None)
        offsets = "embedding/CumSum/output_0"
        self.make_sub("embedding/offsets", [offsets, one], dtype=ir.DataType.INT64, shape=None)
        offsets = "embedding/offsets/output_0"
        self.make_shape("embedding/ids_shape", ids, shape=None)
        ids_shape = "embedding/ids_shape/output_0"
        self.make_reshape("embedding/indices", [offsets, ids_shape], dtype=ir.DataType.INT64, shape=None)
        offsets = "embedding/indices/output_0"
        self.make_shape("embedding/feature_shape", features, shape=None)
        feature_shape = "embedding/feature_shape/output_0"
        self.make_gather("embedding/dummy_index", [feature_shape, zero], dtype=ir.DataType.INT64, shape=None, axis=0)
        dummy_index = "embedding/dummy_index/output_0"
        self.make_where("embedding/gather_indices", [mask, offsets, dummy_index], dtype=ir.DataType.INT64, shape=None)
        indices = "embedding/gather_indices/output_0"
        self.make_initializer(torch.zeros(1, hidden_size), "embedding/dummy", to=self.io_dtype)
        dummy = "embedding/dummy"
        self.make_concat("embedding/padded_features", [features, dummy], dtype=self.io_dtype, shape=None, axis=0)
        padded = "embedding/padded_features/output_0"
        self.make_gather("embedding/image_embeddings", [padded, indices], dtype=self.io_dtype, shape=None, axis=0)
        image_embeds = "embedding/image_embeddings/output_0"
        axes = "/model/constants/INT64/[-1]"
        self.make_unsqueeze("embedding/broadcast_mask", [mask, axes], dtype=ir.DataType.BOOL, shape=None)
        mask = "embedding/broadcast_mask/output_0"
        self.make_where("embedding/Where", [mask, image_embeds, embedded], dtype=self.io_dtype, shape=None)
        merged = "embedding/Where/output_0"
        self.rename_value(merged, "inputs_embeds")
        self.model.graph.outputs.append(
            self.make_value("inputs_embeds", self.io_dtype, ["batch_size", "sequence_length", hidden_size])
        )
        del self.weights
