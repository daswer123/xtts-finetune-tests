import os
import wandb
import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from TTS.tts.datasets import load_tts_samples
from TTS.config.shared_configs import BaseDatasetConfig

from libs.discrete_vae import DiscreteVAE
from libs.utils import TorchMelSpectrogram
from libs.dvae_dataset import DVAEDataset

def load_custom_samples(dataset_path):
    train_samples = []
    eval_samples = []

    with open(os.path.join(dataset_path, 'metadata_train.txt'), 'r') as f:
        for line in f:
            sample = {'audio_file': line.strip()}
            train_samples.append(sample)

    with open(os.path.join(dataset_path, 'metadata_eval.txt'), 'r') as f:
        for line in f:
            sample = {'audio_file': line.strip()}
            eval_samples.append(sample)

    return train_samples, eval_samples

def train_dvae(dvae_checkpoint, mel_norm_file, dataset_path, epochs=20, batch_size=3, learning_rate=5e-05, use_custom_dataset=False):
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

    opt = Adam(dvae.parameters(), lr=learning_rate)
    torch_mel_spectrogram_dvae = TorchMelSpectrogram(
        mel_norm_file=mel_norm_file, sampling_rate=22050
    ).cuda()

    if use_custom_dataset:
        train_samples, eval_samples = load_custom_samples(dataset_path)
    else:
        config_dataset = BaseDatasetConfig(
            formatter="ljspeech",
            dataset_name="ljspeech",
            path=dataset_path,
            meta_file_train=f"{dataset_path}/metadata_norm.txt",
            language="en",
        )

        DATASETS_CONFIG_LIST = [config_dataset]
        train_samples, eval_samples = load_tts_samples(
            DATASETS_CONFIG_LIST,
            eval_split=True,
            eval_split_max_size=256,
            eval_split_size=0.01,
        )

    eval_dataset = DVAEDataset(eval_samples, 22050, True)
    train_dataset = DVAEDataset(train_samples, 22050, False)

    eval_data_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=eval_dataset.collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=4,
        pin_memory=False,
    )

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
            batch['mel'] = torch_mel_spectrogram_dvae(batch['wav'])
            remainder = batch['mel'].shape[-1] % 4
            if remainder:
                batch['mel'] = batch['mel'][:, :, :-remainder]
        except NotImplementedError:
            pass
        return batch

    for epoch in range(epochs):
        for cur_step, batch in enumerate(train_data_loader):
            opt.zero_grad()
            batch = format_batch(batch)
            recon_loss, commitment_loss, out = dvae(batch['mel'])
            total_loss = recon_loss + commitment_loss
            total_loss.backward()
            clip_grad_norm_(dvae.parameters(), 0.5)
            opt.step()

            log = {'epoch': epoch, 'cur_step': cur_step, 'loss': total_loss.item(), 'recon_loss': recon_loss.item(), 'commit_loss': commitment_loss.item()}
            print(f"Epoch: {epoch}, Step: {cur_step}, Loss: {total_loss.item()}, Recon Loss: {recon_loss.item()}, Commit Loss: {commitment_loss.item()}")
            wandb.log(log)
            torch.cuda.empty_cache()

if __name__== "__main__":
    dvae_checkpoint = 'path/to/dvae_checkpoint.pth'
    mel_norm_file = 'path/to/mel_stats.pth'
    dataset_path = 'path/to/dataset'
    use_custom_dataset = True  # Set to False if you want to use the default dataset

    train_dvae(dvae_checkpoint, mel_norm_file, dataset_path, use_custom_dataset=use_custom_dataset)
