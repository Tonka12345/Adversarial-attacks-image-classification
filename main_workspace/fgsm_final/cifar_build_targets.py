from torchvision.datasets import CIFAR100
print("CIFAR100 imported from:", CIFAR100)


# fine label index -> coarse label index
fine_to_coarse = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13
]

def build_target_vector_same_superclass():
    # napravi dict coarse_label -> fine_label list
    coarse_to_fine = {}
    for fine, coarse in enumerate(fine_to_coarse):
        coarse_to_fine.setdefault(coarse, []).append(fine)

    target = [0] * 100
    for fine in range(100):
        coarse = fine_to_coarse[fine]
        alternatives = [f for f in coarse_to_fine[coarse] if f != fine]
        # uzmi prvi iz iste coarse klase koji nije sam sebe
        target[fine] = alternatives[0] if alternatives else fine
    return target

def build_target_vector_different_superclass():
    coarse_to_fine = {}
    for fine, coarse in enumerate(fine_to_coarse):
        coarse_to_fine.setdefault(coarse, []).append(fine)

    all_coarse = set(coarse_to_fine.keys())

    target = [0] * 100
    for fine in range(100):
        coarse = fine_to_coarse[fine]
        # nađi prvi fine label iz bilo koje druge coarse klase
        for other_coarse in all_coarse:
            if other_coarse != coarse:
                target[fine] = coarse_to_fine[other_coarse][0]
                break
    return target

