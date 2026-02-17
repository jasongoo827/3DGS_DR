import numpy as np

import torch
import torch.nn as nn

from torch.cuda.amp import custom_fwd, custom_bwd

# why try-catch? ImportError 정말 일어남?
from _cubemapencoder import _backend

class _cubemap_encode(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(inputs, embeddings, fail_value):
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

    # PyTorch 2.x 방식
    @staticmethod
    def setup_context(ctx, inputs, output):

        # 컨텍스트 설정 (forward의 입력/출력을 받아 필요한 것만 저장)
        # inputs: (inputs, embeddings, fail_value)
        inp, embeddings, fail_value = inputs
        ctx.save_for_backward(inp, embeddings)

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
        super().__init()

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