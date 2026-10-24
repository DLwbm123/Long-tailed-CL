"""Known source-cell correspondences; reflected cells are never positive matches."""
import random


def matching_cells(first_shift, second_shift, side=14):
    def cells(shift):
        dy, dx = shift
        assert dy in (-1, 0, 1) and dx in (-1, 0, 1)
        return {(r-dy, c-dx): r*side+c for r in range(side) for c in range(side)
                if 0 <= r-dy < side and 0 <= c-dx < side}
    first, second = cells(first_shift), cells(second_shift)
    common = sorted(first.keys() & second.keys())
    return [first[k] for k in common], [second[k] for k in common]


def view_receipt(seed, count):
    rng = random.Random(seed)
    shifts = [tuple(rng.choice((-1, 0, 1)) for _ in range(2)) for _ in range(2)]
    photometric = [[(rng.uniform(.9, 1.1), rng.uniform(-.05, .05)) for _ in range(count)] for _ in range(2)]
    first, second = matching_cells(*shifts)
    return dict(shifts=shifts, photometric=photometric, first_cells=first, second_cells=second)


def augment(resized_images, receipt):
    import torch
    from torch.nn import functional as F
    assert resized_images.ndim == 4 and resized_images.shape[1:] == (3, 224, 224)
    result = []
    padded = F.pad(resized_images, (16, 16, 16, 16), mode='reflect')
    for shift, photo in zip(receipt['shifts'], receipt['photometric']):
        dy, dx = shift
        value = padded[:, :, 16-16*dy:240-16*dy, 16-16*dx:240-16*dx]
        params = torch.tensor(photo, dtype=value.dtype, device=value.device)
        value = (value * params[:, 0, None, None, None] + params[:, 1, None, None, None]).clamp(0, 1)
        result.append((value - .5) / .5)
    return result
