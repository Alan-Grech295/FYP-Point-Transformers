import torch
from torch import nn


def print_gradient_norm(model: nn.Module):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            # You can also check individual parameter norms:
            # if param_norm > threshold:
            #     print(f"Exploding gradient for parameter: {p.name}, norm: {param_norm}")
    total_norm = total_norm ** 0.5
    print(f"Total Gradient Norm: {total_norm:.4f}")


def print_gradient_norm_named(model: nn.Module):
    total_norm = 0
    params = []
    param_names = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            params.append(param_norm.item())
            param_names.append(name)
            total_norm += param_norm.item() ** 2
            # You can also check individual parameter norms:
            # if param_norm > 10:
            #     print(f"Exploding gradient for parameter: {name}, norm: {param_norm}")
    total_norm = total_norm ** 0.5
    print(f"Total Gradient Norm: {total_norm:.4f}")
    params = torch.tensor(params)
    min_grad_index = torch.argmin(params)
    max_grad_index = torch.argmax(params)
    print(
        f"Min gradient norm: {param_names[min_grad_index]} - {params[min_grad_index]}, Max gradient norm: {param_names[max_grad_index]} - {params[max_grad_index]}")
