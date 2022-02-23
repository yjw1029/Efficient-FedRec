#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import List, Optional

import torch
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_sequence


def _gen_packed_data(
    minibatch_size: int,
    max_seq_length: int,
    input_dim: int,
    batch_first: bool,
    sorted_: Optional[bool] = False,
) -> PackedSequence:
    """
    This is used to generate random PackedSequence data, sampled from a normal distribution, for testing DPLSTM.

    Args:
        minibatch_size : Total number of sequences to generate
        max_seq_length : The maximum number of timesteps of a sequence
        input_dim : The embedding dimension of a sequence at any timestep
        batch_first : If this is true, data is first generated using a padded sequence of dimension (minibatch_size x max_seq_len x input_dim) , else: (max_seq_length x minibatch_size x input_dim)
        sorted_ : If this is true then the original generated data used to produce the PackedSequence will already be ordered based on sequence lengths, else a random order and the 'sorted_indices'
                    and 'unsorted_indices' fields will be None.

    Return Value:
        packed_data : A PackedSequence object with its data sampled from a normal distribution.
    """

    if batch_first:
        data = []
        seq_lengths = []
        for _ in range(minibatch_size):
            seq_length = torch.randint(1, max_seq_length + 1, (1,)).item()
            seq_lengths.append(seq_length)
            data.append(torch.randn(seq_length, input_dim))

        if sorted_:
            data = sorted(data, key=lambda x: x.shape[0], reverse=True)
            seq_lengths = sorted(seq_lengths, reverse=True)
            packed_data = pack_padded_sequence(
                pad_sequence(data, batch_first=True),
                seq_lengths,
                batch_first=True,
                enforce_sorted=True,
            )
        else:
            packed_data = pack_padded_sequence(
                pad_sequence(data, batch_first=True),
                seq_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
    else:
        seq_lengths = [
            torch.randint(1, max_seq_length + 1, (1,)).item()
            for _ in range(minibatch_size)
        ]
        if sorted_:
            seq_lengths = sorted(seq_lengths, reverse=True)
        padded_data = torch.zeros((max_seq_length, minibatch_size, input_dim))
        for i in range(minibatch_size):
            padded_data[: seq_lengths[i], i, :] = torch.randn(seq_lengths[i], input_dim)

        if sorted_:
            packed_data = pack_padded_sequence(
                padded_data, seq_lengths, batch_first=False, enforce_sorted=True
            )
        else:
            packed_data = pack_padded_sequence(
                padded_data, seq_lengths, batch_first=False, enforce_sorted=False
            )

    return packed_data


def compute_seq_lengths(batch_sizes: torch.Tensor) -> List[int]:
    """
    Computes the sequence lengths of a PackedSequence represented with batch_sizes.

    Args:
        batch_sizes: Contains the batch sizes as stored in a PackedSequence

    Returns:
        running_seq_lengths: the length parameter used in the torch.nn.utils.rnn.packed_padded_sequence function
        to create a PackedSequence. It's a list of the same length as batch_sizes.
    """

    max_batch_size = batch_sizes[0]
    if len(batch_sizes) == 1:
        return [1] * max_batch_size

    running_seq = 0
    running_seq_lengths = []
    for i in range(1, len(batch_sizes)):
        delta = batch_sizes[i - 1].item() - batch_sizes[i].item()
        running_seq += 1
        running_seq_lengths += delta * [running_seq]

    running_seq += 1
    running_seq_lengths += batch_sizes[-1].item() * [running_seq]
    running_seq_lengths.reverse()
    return running_seq_lengths
