import os
import glob
import ffmpeg
import argparse
from concurrent.futures import ThreadPoolExecutor

def split_audio_file(audio_file, output_dir, sample_rate=22050, max_wav_len=11):
    """
    Splits the audio file into chunks of max_wav_len seconds and saves them to the output directory.
    """
    print(f'Splitting {audio_file}')

    # Skip audio files with .json extension
    if audio_file.endswith(".json"):
        return

    audio_info = ffmpeg.probe(audio_file)
    duration = float(audio_info['format']['duration'])

    for i in range(0, int(duration), max_wav_len):
        output_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_{i:04d}.wav")

        stream = ffmpeg.input(audio_file, ss=i, t=max_wav_len)
        stream = ffmpeg.output(stream, output_file, ar=sample_rate)
        ffmpeg.run(stream)

def process_audio_files(input_dir, output_dir, sample_rate=22050, max_wav_len=11):
    """
    Processes audio files from the input directory, splits them into chunks of max_wav_len seconds,
    and saves them to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_files = glob.glob(os.path.join(input_dir, '*.*'))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(split_audio_file, audio_file, output_dir, sample_rate, max_wav_len) for audio_file in audio_files]

        for future in futures:
            future.result()
            print(f'Processed {future}')

def create_metadata_files(input_dir, output_dir, train_percent=0.8):
    """
    Creates metadata files (metadata_train.txt and metadata_eval.txt)
    from the processed audio files in the output directory.
    """
    audio_files = glob.glob(os.path.join(input_dir, '*.wav'))

    train_metadata = os.path.join(output_dir, 'metadata_train.txt')
    eval_metadata = os.path.join(output_dir, 'metadata_eval.txt')

    train_files = audio_files[:int(len(audio_files) *train_percent)]
    eval_files = audio_files[int(len(audio_files)* train_percent):]

    with open(train_metadata, 'w') as f:
        f.write('\n'.join(train_files))

    with open(eval_metadata, 'w') as f:
        f.write('\n'.join(eval_files))

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Process audio files for training.')
    parser.add_argument('--input_data', type=str, required=True, help='Path to the input data directory')
    parser.add_argument('--output_path', type=str, default='processed_dataset', help='Path to the output directory')
    parser.add_argument('--train_percent', type=float, default=0.8, help='Percentage of data for training')
    parser.add_argument('--max_audio_length', type=int, default=11, help='Maximum length of audio chunks in seconds')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Sample rate of audio files')

    args = parser.parse_args()

    process_audio_files(args.input_data, args.output_path, sample_rate=args.sample_rate, max_wav_len=args.max_audio_length)
    create_metadata_files(args.output_path, args.output_path, train_percent=args.train_percent)
