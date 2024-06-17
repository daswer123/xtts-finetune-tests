import os
import glob
import json
import ffmpeg
from concurrent.futures import ThreadPoolExecutor

def split_audio_file(audio_file, output_dir, sample_rate=22050, max_wav_len=11):
    """
    Split audio file into chunks of max_wav_len seconds and save them to output directory.
    """
    print(f'Splitting {audio_file}')

    # if audio file .json skip
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

def create_metadata_files(input_dir, output_dir, language):
    """
    Create metadata files (metadata_train_<language>.txt and metadata_eval_<language>.txt)
    from processed audio files in the output directory.
    """
    audio_files = glob.glob(os.path.join(input_dir, '*.wav'))

    train_metadata = os.path.join(output_dir, f'metadata_train_{language}.txt')
    eval_metadata = os.path.join(output_dir, f'metadata_eval_{language}.txt')

    train_files = audio_files[:int(len(audio_files)*0.8)]
    eval_files = audio_files[int(len(audio_files)*0.8):]

    with open(train_metadata,'w') as f:
        f.write('\n'.join(train_files))

    with open(eval_metadata,'w') as f:
        f.write('\n'.join(eval_files))

def copy_config_file(input_dir,input_output):
     """
     Copy config.json file from input directory to output directory.
     """
     input_config=os.path.join(input_dir,'config.json')
     output_config=os.path.join(input_output,'config.json')

     with open(input_config,'r') as input_file ,open(output_config,'w') as output_file :
         config=json.load(input_file)
         json.dump(config ,input_output)

if __name__== "__main__":
     input_directory='dataset_raw'
     output_directory='dataset_ready'
     max_wav_len=11

# Read language from config.json in input directory
with open(os.path.join(input_directory ,'config.json'),'r') as f:
      config=json.load(f)
      language=config['language']

process_audio_files(input_directory ,output_directory ,max_wav_len=max_wav_len )
create_metadata_files(output_directory ,output_directory ,language )
copy_config_file(input_directory ,output_directory )
