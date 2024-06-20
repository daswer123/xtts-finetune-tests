# XTTS Fine-tuning Repository

In this repository, I will be posting code that I think can be used to fine-tune various parts of XTTS.

I'm an enthusiast and just testing and trying, so don't expect everything to work perfectly. 😄

## DVAE - ✅ Complete

The development of DVAE fine-tuning scripts is completed. You can find more details and instructions in the [dvae-finetune](https://github.com/daswer123/xtts-finetune-tests/tree/main/dvae-finetune) directory.

The basic idea for DVAE fine-tuning is taken from this [GitHub issue](https://github.com/coqui-ai/TTS/issues/3704).

### Example

![DVAE Fine-tuning Example](https://github.com/daswer123/xtts-finetune-tests/assets/22278673/e99e4628-6b2e-414a-ab5b-9a9f72a5049f)

## GPT-2 ( XTTS Encoder ) - 🚧 Work in Progress

The original training recipe for GPT-2 fine-tuning can be found in the [xtts-finetune-webui](https://github.com/daswer123/xtts-finetune-webui) repository.

Currently, I'm working on incorporating the following suggestions:

- Improving speaker conditioning for out-of-training data samples/speakers.
- Modifying the training recipe to make the model more robust.
- Exploring the use of different spoken content while keeping the speaker characteristics the same during training.

The goal is to enhance the model's ability to capture speaker style and improve performance on out-of-distribution samples.

## HifiGAN ( XTTS Decoder ) - ❓ No Info

At the moment, there is no specific information or code available for fine-tuning the HifiGAN component of XTTS.

## Getting Started

To get started with XTTS fine-tuning, please refer to the individual directories for each component. Each directory contains a separate README file with specific instructions, requirements, and examples for fine-tuning the respective component.

Feel free to explore, experiment, and contribute to this repository. If you have any questions or suggestions, don't hesitate to reach out.

Happy fine-tuning! 😊