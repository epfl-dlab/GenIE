import numpy as np
import torch
import pickle


def label_smoothed_nll_loss(
    lprobs: torch.Tensor,
    target: torch.Tensor,
    target_attention_mask: torch.Tensor,
    epsilon: float,
    ignore_index: int = None,
    reduce: bool = True,
):
    # target.shape -> batch_size x tgt_seq_length; lprobs.shape -> batch_size x tgt_seq_length x vocabulary_size
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)  # target.shape -> batch_size x tgt_seq_length x 1

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target.clamp_min_(0)

        nll_loss = -lprobs.gather(dim=-1, index=target)  # get the log prob terms corresponding to the target indices
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)  # calculations needed for the smoothed loss

        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        # TODO: make sure that the dimensions here match i.e. do you actually want the squeeze??
        # TODO: If anywhere, shouldn't it be outside? (see below)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

        # nll_loss = nll_loss.squeeze(-1)
        # smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.squeeze(-1)
    smooth_loss = smooth_loss.squeeze(-1)

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

    # TODO: Why is "the number of classes" (lprobs.size(-1) - 1)? What is the -1 for? Is it padding?
    eps_i = epsilon / (lprobs.size(-1) - 1)
    # TODO: Check correctness. shouldn't it be loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    # loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss

    # Normalize the loss by diving with the number of non-padded
    num_tokens = target_attention_mask.sum()
    loss, nll_loss = loss / num_tokens, nll_loss / num_tokens

    return loss, nll_loss


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def chunk_it(seq, num=1):
    assert num > 0
    chunk_len = len(seq) // num
    chunks = [seq[i * chunk_len : i * chunk_len + chunk_len] for i in range(num)]

    diff = len(seq) - chunk_len * num
    for i in range(diff):
        chunks[i].append(seq[chunk_len * num + i])

    return chunks
