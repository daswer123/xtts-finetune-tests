import os
import json
import argparse
import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from TTS.tts.layers.xtts.dvae import DiscreteVAE
from utils.utils import TorchMelSpectrogram
from utils.dvae_dataset import DVAEDataset

def load_custom_dataset(dataset_path, language):
    metadata_train_file = os.path.join(dataset_path, f'metadata_train.txt')
    metadata_eval_file = os.path.join(dataset_path, f'metadata_eval.txt')

    with open(metadata_train_file, 'r') as f:
        train_samples = [{'audio_file': line.strip(), 'language': language} for line in f]

    with open(metadata_eval_file, 'r') as f:
        eval_samples = [{'audio_file': line.strip(), 'language': language} for line in f]

    return train_samples, eval_samples

def train_dvae(args):
    """Train DVAE model on custom dataset."""

    # Step 1: Load DVAE model
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
    dvae.load_state_dict(torch.load(args.dvae_checkpoint), strict=False)
    dvae.cuda()

    # Use mixed precision if enabled
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_mixed_precision)

    # Step 2: Set up optimizer and mel spectrogram converter
    opt = Adam(dvae.parameters(), lr=args.learning_rate)
    torch_mel_spectrogram_dvae = TorchMelSpectrogram(
        mel_norm_file=args.mel_norm_file, sampling_rate=22050
    ).cuda()

    # Step 3: Load dataset
    train_samples, eval_samples = load_custom_dataset(args.dataset_path, args.language)

    eval_dataset = DVAEDataset(eval_samples, 22050, True)
    train_dataset = DVAEDataset(train_samples, 22050, False)

    eval_data_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=eval_dataset.collate_fn,
        num_workers=args.num_workers // 2,
        pin_memory=True,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Step 4: Set up training with gradient accumulation and profiling (if needed)
    accumulation_steps = max(1, args.batch_size // args.batch_size)

    torch.set_grad_enabled(True)
    dvae.train()

    if args.use_wandb:
        import wandb
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

    # Step 5: Run training loop
    best_loss = float('inf')
    best_epoch = -1
    total_steps = len(train_data_loader)*args.epochs

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_commit_loss = 0

        for cur_step, batch in enumerate(train_data_loader):
            opt.zero_grad()
            batch = format_batch(batch)

            with torch.cuda.amp.autocast(enabled=args.use_mixed_precision):
                recon_loss, commitment_loss, out = dvae(batch['mel'])

                recon_loss = recon_loss.mean()
                commitment_loss = commitment_loss.mean()
                total_loss = recon_loss + commitment_loss

            scaler.scale(total_loss).backward()
            clip_grad_norm_(dvae.parameters(), args.grad_clip_norm)

            scaler.step(opt)
            scaler.update()

            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_commit_loss += commitment_loss.item()

            global_step = epoch*len(train_data_loader) + cur_step + 1
            print(f"Epoch: {epoch+1}/{args.epochs}, Step: {global_step}/{total_steps}, Loss: {total_loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, Commit Loss: {commitment_loss.item():.4f}")

            if args.use_wandb:
                wandb.log({
                    'step': global_step,
                    'loss': total_loss.item(),
                    'recon_loss': recon_loss.item(),
                    'commit_loss': commitment_loss.item()
                })

        # Calculate average losses for the epoch
        avg_loss = epoch_loss / len(train_data_loader)
        avg_recon_loss = epoch_recon_loss / len(train_data_loader)
        avg_commit_loss = epoch_commit_loss / len(train_data_loader)

        # Print epoch summary
        print(f"Epoch: {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}, Avg Recon Loss: {avg_recon_loss:.4f}, Avg Commit Loss: {avg_commit_loss:.4f}")

        # Log metrics to wandb (if enabled)
        if args.use_wandb:
            wandb.log({
                'epoch': epoch+1,
                'avg_loss': avg_loss,
                'avg_recon_loss': avg_recon_loss,
                'avg_commit_loss': avg_commit_loss
            })

        # Save best model checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            save_path = f'best_dvae_{args.language}.pth'
            torch.save(dvae.state_dict(), save_path)
            print(f"Saved best model checkpoint at epoch {best_epoch} with loss {best_loss:.4f}")

        # Save model checkpoint every few epochs
        if (epoch + 1) % args.save_every == 0:
            save_path = f'finetuned_dvae_{args.language}_epoch{epoch+1}.pth'
            torch.save(dvae.state_dict(), save_path)
            print(f"Saved model checkpoint at epoch {epoch+1}")

    print(f"Training completed. Best model found at epoch {best_epoch} with loss {best_loss:.4f}")

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Train DVAE model on custom dataset')
    parser.add_argument('--dvae_checkpoint', type=str, default='./base_model/dvae.pth', help='Path to pre-trained DVAE checkpoint')
    parser.add_argument('--mel_norm_file', type=str, default='./base_model/mel_stats.pth', help='Path to mel normalization file')
    parser.add_argument('--dataset_path', type=str, default='./dataset_ready', help='Path to custom dataset')
    parser.add_argument('--language', type=str, required=True, help='Language of the custom dataset')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-05, help='Learning rate for optimizer')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--grad_clip_norm', type=float, default=0.5, help='Gradient clipping norm value')
    parser.add_argument('--use_mixed_precision', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--use_wandb', action='store_true', help='Enable logging to Weights and Biases')
    parser.add_argument('--save_every', type=int, default=10, help='Save model checkpoint every N epochs')

    args = parser.parse_args()
    train_dvae(args)
