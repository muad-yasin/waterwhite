"""The shuffle of a cyclic run, computed as the trainer's sampler draws it (plan 3.5,
the owner's ruling on the order of a cyclic run).

The cyclic sampler with data sharding draws, for each epoch, one permutation of
its rank's bucket from a generator seeded with the epoch number (not with the run's
seed), and holds it in no file. This module makes the same draw by calling torch
the way the sampler does:

    g = torch.Generator()
    g.manual_seed(epoch)
    torch.randperm(bucket, generator=g)

and adds the rank's first index, ``rank * bucket``, to each place of it. It does not
drive the trainer's sampler class, and it does not restate torch's generator in
another library: numpy's permutations equal torch's for no seed and size tried
(docs/join.md section 6). The result counts only when its digest equals the one
the run record holds (``_join._cyclic_orders``).

torch is imported here, inside ``epoch_permutation``, and nowhere else. A join of a
run that needs no shuffle (a sequential run, a cyclic run answered by an earlier
question) never loads it. Where it cannot be imported the registration of a cyclic
run is refused with "torch not available" and nothing is written.
"""

from __future__ import annotations

from ..errors import TorchUnavailable


def epoch_permutation(epoch: int, bucket: int) -> list[int]:
    """The permutation of one rank's bucket in one epoch: ``bucket`` places, drawn
    as the cyclic sampler draws them. The same list serves every rank."""
    try:
        import torch
    except ImportError as error:
        raise TorchUnavailable(
            "a run read with the cyclic sampler needs torch to compute its shuffle, and torch "
            f"cannot be imported here ({error}); nothing was registered. torch is an optional "
            "dependency of the tool, needed for cyclic runs only"
        ) from None
    generator = torch.Generator()
    generator.manual_seed(epoch)
    return torch.randperm(bucket, generator=generator).tolist()


def rank_order(permutation: list[int], rank: int, bucket: int) -> list[int]:
    """One rank's whole order for an epoch: its first index plus each place of
    the permutation. This is the list the run's shuffle digest covers (DECISIONS.md D-16)."""
    first = rank * bucket
    return [first + place for place in permutation]
