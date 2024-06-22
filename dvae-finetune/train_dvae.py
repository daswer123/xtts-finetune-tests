import os
import json
import argparse
import torch
import torchaudio
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from termcolor import colored
from prettytable import PrettyTable
import logging

import bitsandbytes as bnb

from TTS.tts.layers.xtts.dvae import DiscreteVAE
from utils.utils import TorchMelSpectrogram
from utils.dvae_dataset import DVAEDataset

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_custom_dataset(dataset_path, language):
    metadata_train_file = os.path.join(dataset_path, f'metadata_train.txt')
    metadata_eval_file = os.path.join(dataset_path, f'metadata_eval.txt')

    with open(metadata_train_file, 'r') as f:
        train_samples = [{'audio_file': line.strip(), 'language': language} for line in f]

    with open(metadata_eval_file, 'r') as f:
        eval_samples = [{'audio_file': line.strip(), 'language': language} for line in f]

    return train_samples, eval_samples

def precompute_mel_spectrograms(dataset, output_dir, torch_mel_spectrogram):
    os.makedirs(output_dir, exist_ok=True)

    for sample in tqdm(dataset, desc="Computing mel-spectrograms"):
        audio_file = sample['audio_file']
        wav, sr = torchaudio.load(audio_file)
        if wav.dim() > 2:
            wav = wav.mean(dim=0)
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)
        mel = torch_mel_spectrogram(wav)


        mel_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}.pt")
        torch.save(mel, mel_file)

def train_dvae(args):
    """Train DVAE model on custom dataset."""

    logger = setup_logging("train_log.log")

    # Step 0: Create train folder if not exists
    if not os.path.exists("train"):
        os.makedirs("train")

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
    opt = bnb.optim.Adam8bit(dvae.parameters(), lr=args.learning_rate)
    torch_mel_spectrogram = TorchMelSpectrogram(
        mel_norm_file=args.mel_norm_file, sampling_rate=22050
    ).cuda()

    # Step 3: Load dataset
    train_samples, eval_samples = load_custom_dataset(args.dataset_path, args.language)

    # Step 4: Precompute mel-spectrograms
    train_mels_dir = os.path.join(args.dataset_path, "mels", "train")
    eval_mels_dir = os.path.join(args.dataset_path, "mels", "eval")
    precompute_mel_spectrograms(train_samples, train_mels_dir, torch_mel_spectrogram)
    precompute_mel_spectrograms(eval_samples, eval_mels_dir, torch_mel_spectrogram)

    eval_dataset = DVAEDataset(eval_samples, eval_mels_dir, 22050, True)
    train_dataset = DVAEDataset(train_samples, train_mels_dir, 22050, False)

    eval_data_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=eval_dataset.collate_fn,
        num_workers=args.num_workers // 2,
        pin_memory=True,
        persistent_workers=True
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # Step 5: Set up training
    torch.set_grad_enabled(True)
    dvae.train()

    if args.use_wandb:
        import wandb
        wandb.init(project='train_dvae')
        wandb.watch(dvae)

    # Step 6: Run training loop
    best_metrics = {
        'loss': float('inf'),
        'recon_loss': float('inf'),
        'commit_loss': float('inf'),
        'epoch': -1
    }
    total_steps = len(train_data_loader) * args.epochs

    # Create a table to display metrics
    metrics_table = PrettyTable()
    metrics_table.field_names = ["Epoch", "Avg Loss", "Avg Recon Loss", "Avg Commit Loss"]

    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_commit_loss = 0

        progress_bar = tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for cur_step, batch in enumerate(progress_bar):
            opt.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.use_mixed_precision):
                recon_loss, commitment_loss, out = dvae(batch['mel'].cuda())

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

            global_step += 1
            progress_bar.set_postfix(step=f"{cur_step+1}/{len(train_data_loader)}", global_step=global_step, loss=total_loss.item(), recon_loss=recon_loss.item(), commit_loss=commitment_loss.item())

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

        # Update the best metrics if the current epoch is better
        if avg_loss < best_metrics['loss']:
            best_metrics['loss'] = avg_loss
            best_metrics['recon_loss'] = avg_recon_loss
            best_metrics['commit_loss'] = avg_commit_loss
            best_metrics['epoch'] = epoch + 1

        # Add metrics to the table
        metrics_row = [epoch+1, f"{avg_loss:.4f}", f"{avg_recon_loss:.4f}", f"{avg_commit_loss:.4f}"]

        # Compare with the best metrics and highlight the differences
        if epoch + 1 == best_metrics['epoch']:
            metrics_row[1] = colored(metrics_row[1], 'green')
            metrics_row[2] = colored(metrics_row[2], 'green')
            metrics_row[3] = colored(metrics_row[3], 'green')
        else:
            if avg_loss < best_metrics['loss']:
                metrics_row[1] = colored(metrics_row[1], 'green')
            else:
                metrics_row[1] = colored(metrics_row[1], 'white')

            if avg_recon_loss < best_metrics['recon_loss']:
                metrics_row[2] = colored(metrics_row[2], 'green')
            else:
                metrics_row[2] = colored(metrics_row[2], 'white')

            if avg_commit_loss < best_metrics['commit_loss']:
                metrics_row[3] = colored(metrics_row[3], 'green')
            else:
                metrics_row[3] = colored(metrics_row[3], 'white')

        metrics_table.add_row(metrics_row)

        # Save the best model checkpoint
        if avg_loss < best_metrics['loss']:
            save_path = f'train/best_dvae_{args.language}.pth'
            torch.save(dvae, save_path)
            logger.info(f"Saved best model checkpoint at epoch {best_metrics['epoch']} with loss {best_metrics['loss']:.4f}")

        # Log metrics to wandb (if enabled)
        if args.use_wandb:
            wandb.log({
                'epoch': epoch+1,
                'avg_loss': avg_loss,
                'avg_recon_loss': avg_recon_loss,
                'avg_commit_loss': avg_commit_loss
            })

        # Save model checkpoint every few epochs
        if (epoch + 1) % args.save_every == 0:
            save_path = f'train/finetuned_dvae_{args.language}_epoch{epoch+1}.pth'
            torch.save(dvae, save_path)
            logger.info(f"Saved model checkpoint at epoch {epoch+1}")

        # Log the metrics table after each epoch
        logger.info("\nTraining metrics:")
        logger.info(str(metrics_table))

    logger.info(f"\nBest metrics achieved at epoch {best_metrics['epoch']}:")
    logger.info(f"Loss: {best_metrics['loss']:.4f}")
    logger.info(f"Reconstruction Loss: {best_metrics['recon_loss']:.4f}")
    logger.info(f"Commitment Loss: {best_metrics['commit_loss']:.4f}")


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Train DVAE model on custom dataset')
    parser.add_argument('--dvae_checkpoint', type=str, default='./base_model/dvae.pth', help='Path to pre-trained DVAE checkpoint')
    parser.add_argument('--mel_norm_file', type=str, default='./base_model/mel_stats.pth', help='Path to mel normalization file')
    parser.add_argument('--dataset_path', type=str, default='./processed_dataset', help='Path to custom dataset')
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
