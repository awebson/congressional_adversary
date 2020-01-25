import torch
from tqdm import tqdm

from bill_decomposer import DecomposerExperiment, DecomposerConfig

torch.manual_seed(42)

device = 'cuda:1'
out_dir = '../results/affine/L4 {}d 10b'
# deltas = (-0.05, -0.1, -0.2, -0.5, -1)
deltas = (-2,)
gamma = 1
beta = 10
batch_size = 64
learning_rate = 1e-4
num_epochs = 50

def main() -> None:
    for delta in tqdm(deltas, desc='Hyperparameter Search'):
        config = DecomposerConfig(
            output_dir=out_dir.format(delta),
            deno_delta=delta,
            deno_gamma=gamma,
            beta=beta,
            architecture='L4',
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            print_stats=None)
        with DecomposerExperiment(config) as auto_save_wrapped:
            auto_save_wrapped.train()

if __name__ == '__main__':
    main()
