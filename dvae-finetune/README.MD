# Fine-tuning DVAE for XTTS

Thanks to @smallsudarshan for providing the code! Of course, I had to work hard to put everything together and adapt it, but I'm glad this person shared their code.

## Introduction

This project is intended for fine-tuning the Discrete VAE (DVAE) model for use with XTTS. This guide outlines the steps to set up and run the training process.

## Requirements

1. **Download the original DVAE and mel_stats:**
   - [mel_stats.pth](https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth?download=true)
   - [dvae.pth](https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth?download=true)

   Place them in the `base_model` folder.

2.**Create a virtual environment:**
   - Tested on Python 3.10.9 and CUDA 12.1.

3.**Ensure ffmpeg is installed on your system:**
   ```sh
   choco install ffmpeg
   ```

4.**Activate your virtual environment and install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

5.**Install PyTorch with CUDA support:**
   ```sh
   pip install torch==2.1.1+cu121 torchaudio==2.1.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

6.**Congratulations! Installation is complete!**

## Dataset Preparation

I made a custom solution for dataset preparation, so keep that in mind.

### Steps:

1. Create a folder with your material:
    - All audio files should be in one folder.
    - Inside this folder, create a file named `config.json` with the following content:
      ```json
      {
          "language": "en"
      }
      ```
      Replace `"en"` with your language code.

2. Run the dataset creation script:
    ```sh
    python dataset_creator.py
    ```

**Note:** Sometimes the script may throw an error, so copy the `config.json` file to the ready dataset folder if you encounter an error during formation.

## Configuration and Training Execution

After you have prepared your dataset:

1. Open the tuner file (`dvae_tunner.py`) and configure all settings.
2. Specify the path to your dataset.
3. Run the script using:
    ```sh
    python train_dvae.py
    ```

Now you are ready to train your DVAE model!
