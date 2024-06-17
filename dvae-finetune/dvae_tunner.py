import os
import json
import wandb
import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from TTS.tts.layers.xtts.dvae import DiscreteVAE
from libs.utils import TorchMelSpectrogram
from libs.dvae_dataset import DVAEDataset

# Define hyperparameters and constants for optimization
DVAE_CHECKPOINT = './base_model/dvae.pth'
MEL_NORM_FILE = './base_model/mel_stats.pth'
DATASET_PATH = './dataset_ready'
EPOCHS = 20
BATCH_SIZE = 10
LEARNING_RATE = 5e-05

NUM_WORKERS = 8  # Number of workers for data loading
GRAD_CLIP_NORM = 0.5  # Gradient clipping norm value

# Enable mixed precision training if available
USE_MIXED_PRECISION = True if torch.cuda.is_available() else False

def load_custom_dataset(dataset_path, language):
    metadata_train_file = os.path.join(dataset_path, f'metadata_train_{language}.txt')
    metadata_eval_file = os.path.join(dataset_path, f'metadata_eval_{language}.txt')

    with open(metadata_train_file, 'r') as f:
        train_samples = [{'audio_file': line.strip(), 'language': language} for line in f]

    with open(metadata_eval_file, 'r') as f:
        eval_samples = [{'audio_file': line.strip(), 'language': language} for line in f]

    return train_samples, eval_samples

def train_dvae(dvae_checkpoint, mel_norm_file, dataset_path, epochs=20, batch_size=3, learning_rate=5e-05):
    """Train DVAE model on custom dataset."""

    # Step 1: Load language from config.json
    with open(os.path.join(dataset_path, 'config.json'), 'r') as f:
        config = json.load(f)
    language = config['language']

    # Step 2: Load DVAE model
    dvae = DiscreteVAE(
        channels=80,
        normalization=None,
        positional_dims=1,
        num_tokens=1024,
        codebook_dim=512,
        hidden_dim=512,
        num_resnet_blocks=3,
        kernel_size=3,
        num_layers=2,
        use_transposed_convs=False,
    )
    dvae.load_state_dict(torch.load(dvae_checkpoint), strict=False)
    dvae.cuda()

    # Use mixed precision if enabled
    scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION)

    # Step 3: Set up optimizer and mel spectrogram converter
    opt = Adam(dvae.parameters(), lr=learning_rate)
    torch_mel_spectrogram_dvae = TorchMelSpectrogram(
        mel_norm_file=mel_norm_file, sampling_rate=22050
    ).cuda()

    # Step 4: Load dataset
    train_samples, eval_samples = load_custom_dataset(dataset_path, language)

    eval_dataset = DVAEDataset(eval_samples, 22050, True)
    train_dataset = DVAEDataset(train_samples, 22050, False)

    eval_data_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=eval_dataset.collate_fn,
        num_workers=NUM_WORKERS // 2,
        pin_memory=True,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Step 5: Set up training with gradient accumulation and profiling (if needed)
    accumulation_steps = max(1, BATCH_SIZE // batch_size)
    
    torch.set_grad_enabled(True)
    dvae.train()
    wandb.init(project='train_dvae')
    wandb.watch(dvae)
    
    def to_cuda(x: torch.Tensor) -> torch.Tensor:
        if x is None:
            return None
        if torch.is_tensor(x):
            x = x.contiguous()
            if torch.cuda.is_available():
                x = x.cuda(non_blocking=True)
        return x
    
    @torch.no_grad()
    def format_batch(batch):
        if isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = to_cuda(v)
        elif isinstance(batch, list):
            batch = [to_cuda(v) for v in batch]
    
        try:
            # Переносим вычисление мел-спектрограммы на GPU
            wavs = to_cuda(batch['wav'])
            batch['mel'] = torch_mel_spectrogram_dvae(wavs)
    
            remainder = batch['mel'].shape[-1] % 4
            if remainder:
                batch['mel'] = batch['mel'][:, :, :-remainder]
        except NotImplementedError:
            pass
        return batch

    # Step 6: Run training loop
    for epoch in range(epochs):
        for cur_step, batch in enumerate(train_data_loader):
            opt.zero_grad()
            batch = format_batch(batch)

            with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
                recon_loss, commitment_loss, out = dvae(batch['mel'])

                recon_loss = recon_loss.mean()
                commitment_loss = commitment_loss.mean()
                total_loss = recon_loss + commitment_loss

            scaler.scale(total_loss).backward()
            clip_grad_norm_(dvae.parameters(), GRAD_CLIP_NORM)

            scaler.step(opt)
            scaler.update()

            log = {'epoch': epoch, 'cur_step': cur_step, 'loss': total_loss.item(), 'recon_loss': recon_loss.item(), 'commit_loss': commitment_loss.item()}

            print(f"Epoch: {epoch}, Step: {cur_step}, Loss: {total_loss.item()}, Recon Loss: {recon_loss.item()}, Commit Loss: {commitment_loss.item()}")

            # Log every 5 epochs
            # if epoch % 5 == 0:
            wandb.log(log)
            # wandb.log(log)

            # torch.cuda.empty_cache()

        # Save finetuned model every few epochs or at the end of training
        if epoch % 10 == 0 or epoch == epochs - 1:
            save_path = f'finetuned_dvae_{language}_epoch{epoch}.pth'
            torch.save(dvae.state_dict(), save_path)

if __name__== "__main__":
    train_dvae(DVAE_CHECKPOINT, MEL_NORM_FILE, DATASET_PATH, EPOCHS, BATCH_SIZE, LEARNING_RATE)
