import os
import glob
import json
import ffmpeg
import argparse
from concurrent.futures import ThreadPoolExecutor

def split_audio_file(audio_file, output_dir, sample_rate=22050, max_wav_len=11):
    """
    Split audio file into chunks of max_wav_len seconds and save them to output directory.
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
    Process audio files from input directory, split them into chunks of max_wav_len seconds,
    and save them to output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_files = glob.glob(os.path.join(input_dir, '*.*'))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(split_audio_file, audio_file, output_dir, sample_rate, max_wav_len) for audio_file in audio_files]

        for future in futures:
            future.result()
            print(f'Processed {future}')

def create_metadata_files(input_dir, output_dir, language, train_percent=0.8, eval_percent=0.2):
    """
    Create metadata files (metadata_train_<language>.txt and metadata_eval_<language>.txt)
    from processed audio files in the output directory.
    """
    audio_files = glob.glob(os.path.join(input_dir, '*.wav'))

    train_metadata = os.path.join(output_dir, f'metadata_train_{language}.txt')
    eval_metadata = os.path.join(output_dir, f'metadata_eval_{language}.txt')

    train_files = audio_files[:int(len(audio_files) *train_percent)]
    eval_files = audio_files[int(len(audio_files)* train_percent):]

    with open(train_metadata, 'w') as f:
        f.write('\n'.join(train_files))

    with open(eval_metadata, 'w') as f:
        f.write('\n'.join(eval_files))

def copy_config_file(input_dir, output_dir):
    """
    Copy config.json file from input directory to output directory.
    """
    input_config = os.path.join(input_dir, 'config.json')
    output_config = os.path.join(output_dir, 'config.json')

    with open(input_config, 'r') as input_file, open(output_config, 'w') as output_file:
        config = json.load(input_file)
        json.dump(config, output_file)

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Process audio files for training.')
    parser.add_argument('--input_data', type=str, required=True, help='Path to input data directory')
    parser.add_argument('--output_path', type=str, default='processed_dataset', help='Path to output directory')
    parser.add_argument('--language', type=str, required=True, help='Language of the audio files')
    parser.add_argument('--train_percent', type=float, default=0.8, help='Percentage of data for training')
    parser.add_argument('--eval_percent', type=float, default=0.2, help='Percentage of data for evaluation')
    parser.add_argument('--max_audio_length', type=int, default=11, help='Maximum length of audio chunks in seconds')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Sample rate of audio files')

    args = parser.parse_args()

    process_audio_files(args.input_data, args.output_path, sample_rate=args.sample_rate, max_wav_len=args.max_audio_length)
    create_metadata_files(args.output_path, args.output_path, args.language, train_percent=args.train_percent, eval_percent=args.eval_percent)
    copy_config_file(args.input_data, args.output_path)
