import torch
import random

from TTS.tts.models.xtts import load_audio

torch.set_num_threads(1)

class DVAEDataset(torch.utils.data.Dataset):
    def __init__(self, samples, sample_rate, is_eval):
        self.sample_rate = sample_rate
        self.is_eval = is_eval
        self.max_wav_len = 255995
        self.samples = samples
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
                _, wav = self.load_item(sample)
            except:
                continue
            if wav is None or (self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len):
                continue
            new_samples.append(sample)
        self.samples = new_samples
        print(" > Total eval samples after filtering:", len(self.samples))

    def load_item(self, sample):
        audiopath = sample["audio_file"]
        wav = load_audio(audiopath, self.sample_rate)
        if wav is None or wav.shape[-1] < (0.5 *self.sample_rate):
            raise ValueError
        return audiopath, wav

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
            audiopath, wav = self.load_item(sample)
        except:
            self.failed_samples.add(sample_id)
            return self[1]

        if wav is None or (self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len):
            self.failed_samples.add(sample_id)
            return self[1]

        res = {
            "wav": wav,
            "wav_lengths": torch.tensor(wav.shape[-1], dtype=torch.long),
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
        batch["wav_lengths"] = torch.stack(batch["wav_lengths"])
        max_wav_len = batch["wav_lengths"].max()
        wav_padded = torch.FloatTensor(B, 1, max_wav_len).zero_()
        for i in range(B):
            wav = batch["wav"][i]
            wav_padded[i, :, : batch["wav_lengths"][i]] = torch.FloatTensor(wav)
        batch["wav"] = wav_padded
        return batch

    def key_samples_by_col(self, samples, col_name):
        samples_by_col = {}
        for sample in samples:
            key = sample[col_name]
            if key not in samples_by_col:
                samples_by_col[key] = []
            samples_by_col[key].append(sample)
        return samples_by_col
