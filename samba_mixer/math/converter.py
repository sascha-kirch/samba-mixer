import torch


def ck_to_soh(ck: torch.Tensor, c0: float) -> torch.Tensor:
    """Converts a provided tensor from its capacity in cycle k to SOH (state of health).

    Args:
        ck (torch.Tensor): Capacities at cycle k
        c0 (float): Rated capacity.

    Returns:
        torch.Tensor: State of Health at cycle k
    """
    return ck * 100 / c0


def soh_to_ck(soh: torch.Tensor, c0: float) -> torch.Tensor:
    """Converts a provided tensor from its SOH (state of health) to capacity at cycle k.

    Args:
        soh (torch.Tensor): SOH at cycle k
        c0 (float): Rated capacity.

    Returns:
        torch.Tensor: Capacities at cycle k
    """
    return soh * c0 / 100
