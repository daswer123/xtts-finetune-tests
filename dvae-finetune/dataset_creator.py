import ffmpeg
import os
import glob

def process_audio_files(input_dir, output_dir, sample_rate=22050):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_files = glob.glob(os.path.join(input_dir, '*.wav'))

    for audio_file in audio_files:
        output_file = os.path.join(output_dir, os.path.basename(audio_file))

        (
            ffmpeg
            .input(audio_file)
            .output(output_file, ar=sample_rate)
            .run(overwrite_output=True)
        )
        print(f'Processed {audio_file} to {output_file}')

if __name__== "__main__":
    input_directory = 'path/to/input_directory'
    output_directory = 'path/to/output_directory'
    process_audio_files(input_directory, output_directory)
