import numpy as np
import torch


# Loss function: MSE + PCC
def pcc_loss(output, target):
    # Convert target to float type
    target = target.float()
    x = output - output.mean(dim=1, keepdim=True)
    y = target - target.mean(dim=1, keepdim=True)

    covariance = (x * y).sum(dim=1)
    bessel_corrected_variance_x = (x ** 2).sum(dim=1)
    bessel_corrected_variance_y = (y ** 2).sum(dim=1)

    pcc = covariance / torch.sqrt(bessel_corrected_variance_x * bessel_corrected_variance_y + 1e-8)
    return 1 - pcc.mean()  # 1 - PCC as the loss


def combined_loss(output, target, alpha=0.5):
    mse = mse_loss_for_block(output, target)
    pcc = pcc_loss(output, target)
    return alpha * pcc + (1 - alpha) * mse


def calculate_pcc(pred, target):
    target = target.float()
    x = pred - pred.mean(dim=1, keepdim=True)
    y = target - target.mean(dim=1, keepdim=True)

    covariance = (x * y).sum(dim=1)
    bessel_corrected_variance_x = (x ** 2).sum(dim=1)
    bessel_corrected_variance_y = (y ** 2).sum(dim=1)

    pcc = covariance / torch.sqrt(bessel_corrected_variance_x * bessel_corrected_variance_y + 1e-8)
    return pcc.mean()


def mse_loss_for_block(pred, target):
    pred = pred.float()
    target = target.float()
    mse_per_sample = ((pred - target) ** 2).mean(dim=1)  # MSE for each node
    avg_mse = mse_per_sample.mean()  # Average MSE for blocks
    return avg_mse


def mae_for_block(pred, target):
    pred = pred.float()
    target = target.float()
    mae_per_sample = (np.abs(pred - target)).mean(dim=1)
    avg_mse = mae_per_sample.mean()
    return avg_mse
