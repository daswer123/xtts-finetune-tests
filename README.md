In this repository I will be posting code that I think can be used to filetune various parts of xtts.

I'm an enthusiast and just testing and trying, don't expect it to work _

# Dvae ( Complete )
## Development is completed scripts are written, more details [here](https://github.com/daswer123/xtts-finetune-tests/tree/main/dvae-finetune)

1. The basic idea is taken from here https://github.com/coqui-ai/TTS/issues/3704

### Example

![image](https://github.com/daswer123/xtts-finetune-tests/assets/22278673/e99e4628-6b2e-414a-ab5b-9a9f72a5049f)

# GPT-2 ( Work in progress )

Original train recipe you can find here: https://github.com/daswer123/xtts-finetune-webui

Now i'm woking on this suggestions

> Few things I have observed:

> Currently the speaker conditioning does not work well for out of training data samples/speakers.
> One of the ways to make the model more robust in this to change the training recipe a bit. Currently the ljspeech data loader completely ignores speaker information.

> During training, the same sample is giving to the perceiver that needs to be synthesized. What if instead, we keep the speaker (and if applicable other characteristics like emotion) the same but use a sample with different spoken content?

> That way, the model might learn that it is the style from the speaker that has to be picked and it might also work a bit better for out-of-distribution (not sure though).

HifiGAN ( No Info )
