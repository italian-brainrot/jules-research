import torch
from torch.optim import Optimizer

class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, encoding_dim))
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_dim, 64),
            torch.nn.ReLU(True),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, input_dim))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class LearnedGradientCompressionOptimizer(Optimizer):
    def __init__(self, params, base_optimizer, encoding_dim=32, chunk_size=256, train_ae_interval=100):
        defaults = dict()
        super(LearnedGradientCompressionOptimizer, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.chunk_size = chunk_size
        self.autoencoder = AutoEncoder(input_dim=chunk_size, encoding_dim=encoding_dim)
        self.autoencoder_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        self.compression_loss = torch.nn.MSELoss()
        self.step_count = 0
        self.train_ae_interval = train_ae_interval

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        all_chunks = []
        param_shapes = []

        # Collect all gradient chunks
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.detach()
                param_shapes.append((p, grad.shape))
                grad_flat = grad.view(-1)

                for i in range(0, grad_flat.shape[0], self.chunk_size):
                    chunk = grad_flat[i:i+self.chunk_size]
                    original_chunk_size = chunk.shape[0]

                    if original_chunk_size < self.chunk_size:
                        padding = torch.zeros(self.chunk_size - original_chunk_size, device=grad.device)
                        chunk_padded = torch.cat([chunk, padding])
                    else:
                        chunk_padded = chunk

                    all_chunks.append(chunk_padded)

        if not all_chunks:
            return loss

        # Batch train the autoencoder
        if self.step_count % self.train_ae_interval == 0:
            chunk_tensor = torch.stack(all_chunks)
            self.autoencoder_optimizer.zero_grad()
            reconstructed_chunks = self.autoencoder(chunk_tensor)
            loss_ae = self.compression_loss(reconstructed_chunks, chunk_tensor)
            loss_ae.backward()
            self.autoencoder_optimizer.step()

        # Reconstruct and apply gradients
        with torch.no_grad():
            chunk_idx = 0
            for p, original_shape in param_shapes:
                grad_flat = p.grad.view(-1)
                reconstructed_chunks_for_param = []

                for i in range(0, grad_flat.shape[0], self.chunk_size):
                    chunk = grad_flat[i:i+self.chunk_size]
                    original_chunk_size = chunk.shape[0]

                    if original_chunk_size < self.chunk_size:
                        padding = torch.zeros(self.chunk_size - original_chunk_size, device=p.grad.device)
                        chunk_padded = torch.cat([chunk, padding])
                    else:
                        chunk_padded = chunk

                    compressed_chunk = self.autoencoder.encoder(chunk_padded)
                    decompressed_chunk_padded = self.autoencoder.decoder(compressed_chunk)

                    if original_chunk_size < self.chunk_size:
                        decompressed_chunk = decompressed_chunk_padded[:original_chunk_size]
                    else:
                        decompressed_chunk = decompressed_chunk_padded

                    reconstructed_chunks_for_param.append(decompressed_chunk)

                reconstructed_grad = torch.cat(reconstructed_chunks_for_param)
                p.grad.data = reconstructed_grad.view(original_shape)

        self.base_optimizer.step()
        self.step_count += 1
        return loss
