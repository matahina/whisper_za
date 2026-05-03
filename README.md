# whisper_za
A python script to use some south african languages with [whisper](https://github.com/openai/whisper)

## Usage

`python3 whisper_za.py [arg1] [arg2] "offline"`

where `[arg1]` is the audio file you want to translate

where `[arg2]` is either `zulu` or `xhosa`

where `[arg3]` is `offline` or empty (if `offline`, it won't check HuggingFace for model and params updates)

***

Check `import` headers for python libs you might need (`transformers`, `pytorch`, `datasets`).
