import os
import torch
import random

class DVAEDataset(torch.utils.data.Dataset):
    def __init__(self, samples, mels_dir, sample_rate, is_eval):
        self.sample_rate = sample_rate
        self.is_eval = is_eval
        self.max_wav_len = 255995
        self.samples = samples
        self.mels_dir = mels_dir
        self.training_seed = 1
        self.failed_samples = set()

        if not is_eval:
            random.seed(self.training_seed)
            random.shuffle(self.samples)
            self.samples = self.key_samples_by_col(self.samples, "language")
            print(" > Sampling by language:", self.samples.keys())
        else:
            self.check_eval_samples()

    def check_eval_samples(self):
        print(" > Filtering invalid eval samples!!")
        new_samples = []
        for sample in self.samples:
            try:
                _, mel = self.load_item(sample)
            except:
                continue
            if mel is None:
                continue
            new_samples.append(sample)
        self.samples = new_samples
        print(" > Total eval samples after filtering:", len(self.samples))

    def load_item(self, sample):
        audiopath = sample["audio_file"]
        mel_file = os.path.join(self.mels_dir, f"{os.path.splitext(os.path.basename(audiopath))[0]}.pt")
        mel = torch.load(mel_file)
        return audiopath, mel

    def __getitem__(self, index):
        if self.is_eval:
            sample = self.samples[index]
            sample_id = str(index)
        else:
            lang = random.choice(list(self.samples.keys()))
            index = random.randint(0, len(self.samples[lang]) - 1)
            sample = self.samples[lang][index]
            sample_id = lang + "_" + str(index)

        if sample_id in self.failed_samples:
            return self[1]

        try:
            audiopath, mel = self.load_item(sample)
        except:
            self.failed_samples.add(sample_id)
            return self[1]

        if mel is None:
            self.failed_samples.add(sample_id)
            return self[1]

        res = {
            "mel": mel,
            "mel_lengths": torch.tensor(mel.shape[-1], dtype=torch.long),
            "filenames": audiopath,
        }
        return res

    def __len__(self):
        if self.is_eval:
            return len(self.samples)
        return sum([len(v) for v in self.samples.values()])

    def collate_fn(self, batch):
        B = len(batch)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}
        batch["mel_lengths"] = torch.stack(batch["mel_lengths"])
        max_mel_len = batch["mel_lengths"].max()
        mel_padded = torch.FloatTensor(B, batch["mel"][0].shape[0], max_mel_len).zero_()
        for i in range(B):
            mel = batch["mel"][i]
            mel_padded[i, :, : batch["mel_lengths"][i]] = mel
        batch["mel"] = mel_padded
        return batch

    def key_samples_by_col(self, samples, col_name):
        samples_by_col = {}
        for sample in samples:
            key = sample[col_name]
            if key not in samples_by_col:
                samples_by_col[key] = []
            samples_by_col[key].append(sample)
        return samples_by_col
