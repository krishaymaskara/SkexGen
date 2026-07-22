# Small constructors for additive attention masks. Allowed locations contain
# zero; blocked locations contain negative infinity before softmax.
import torch


def to_negative_mask(mask):
    # Convert a binary allow/block matrix to Transformer additive-mask values.
    if mask is None:
        return

    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_square_subsequent_mask(sz):
    # Causal mask: each position sees itself and earlier positions only.
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_adj_subsequent_mask(sz):
    # Causal local mask limited to the current and two preceding positions.
    mask = torch.diag(torch.ones(sz), diagonal=0) + torch.diag(torch.ones(sz-1), diagonal=-1)

    if sz >= 2:
        mask = mask + torch.diag(torch.ones(sz-2), diagonal=-2)

    return to_negative_mask(mask)


def generate_adj_mask(sz):
    # Bidirectional local mask spanning up to two neighbors on either side.
    mask = torch.diag(torch.ones(sz), diagonal=0) +\
           torch.diag(torch.ones(sz - 1), diagonal=+1) +\
           torch.diag(torch.ones(sz - 1), diagonal=-1)

    if sz >= 2:
        mask = mask + torch.diag(torch.ones(sz - 2), diagonal=-2) +\
               torch.diag(torch.ones(sz - 2), diagonal=+2)

    return to_negative_mask(mask)
