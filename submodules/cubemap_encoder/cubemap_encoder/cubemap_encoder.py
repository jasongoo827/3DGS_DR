import numpy as np

import torch
import torch.nn as nn

from torch.cuda.amp import custom_fwd, custom_bwd

# why try-catch? ImportError 정말 일어남?
import _cubemapencoder as _backend

class _cubemap_encode(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, embeddings, fail_value):
        ctx.save_for_backward(inputs, embeddings)
        embeddings = embeddings.contiguous()
        inputs = inputs.contiguous()

        # output dimension
        C = embeddings.shape[1]
        # resolution
        L = embeddings.shape[2]
        # Normal 수
        B = inputs.shape[0]

        # C X B 로 넣어야 thread가 메모리 접근 더 빠르게 할 수 있음.
        outputs = torch.empty([C, B], dtype=embeddings.dtype, device=embeddings.device)

        _backend.cubemap_encode_forward(inputs, embeddings, fail_value, outputs,
                                        B, C, L)
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        inputs, embeddings = ctx.saved_tensors
        grad_outputs = grad_outputs.contiguous()

        C = embeddings.shape[1]
        L = embeddings.shape[2]
        B = inputs.shape[0]

        grad_embeddings = torch.zeros_like(embeddings)
        grad_inputs = torch.empty_like(inputs)
        grad_fail =  torch.zeros([C], dtype=embeddings.dtype, device=embeddings.device)

        _backend.cubemap_encode_backward(
            grad_outputs, inputs, embeddings,
            grad_embeddings, grad_inputs, grad_fail,
            B, C, L
        )
        return grad_inputs, grad_embeddings, grad_fail, None, None

cubemap_encode = _cubemap_encode.apply

class CubemapEncoder(nn.Module):
    def __init__(self, output_dim, resolution=256):
        super().__init__()

        self.input_dim = 3
        self.resolution = resolution
        self.output_dim = output_dim

        self.params = nn.ParameterDict({
            'Cubemap_texture': nn.Parameter(torch.rand(6, self.output_dim, resolution, resolution)*10-5), 
            'Cubemap_failv': nn.Parameter(torch.zeros(self.output_dim))
        })

    def forward(self, inputs):
        outputs = cubemap_encode(inputs, self.params['Cubemap_texture'], self.params['Cubemap_failv'])
        # output은 다시 B X C (Normal당 color)
        return outputs.permute(1, 0)