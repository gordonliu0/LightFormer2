from torch.utils.data import WeightedRandomSampler

def create_weighted_sampler(dataset):
    'Weighted samplers help with unbalanced datasets.'
    label_counts = {(1., 0., 1., 0.): 211, (1., 0., 0., 1.): 272, (0., 1., 1., 0.): 43, (0., 1., 0., 1.): 703}
    class_weights = {class_label: 1.0 / count for class_label, count in label_counts.items()}
    sample_weights = [class_weights[tuple(sample["label"].tolist())] for sample in dataset]
    return WeightedRandomSampler(sample_weights, len(sample_weights))
