#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import datetime
import os
from transformers import pipeline
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

try:
    match sys.argv[3]:
        case "offline":
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            print("lalalala")
except:
    pass


match sys.argv[2]:
    case "zulu":
        mod_name = "TheirStory/whisper-medium-zulu"
    case "xhosa":
        mod_name = "TheirStory-Inc/whisper-small-xhosa"

# from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model=mod_name)


# from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained(mod_name)
model = AutoModelForSpeechSeq2Seq.from_pretrained(mod_name)


# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = mod_name

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]


result = pipe(sys.argv[1], return_timestamps=True)
# print(result["text"])
# print(result["chunks"])

vari_p = sys.argv[1].split('.') # parts
res = ['.'.join(vari_p[:-1]), vari_p[-1]]

with open(res[0]+".txt", "w") as f:
     f.write(result["text"])




try:
    os.remove(res[0]+".srt")
except:
    pass


try:
    with open(res[0]+".srt", "a") as f:
        icount = 0
        for elem in result["chunks"]:
            bloup = ""
            for subelem in elem:
                if isinstance(elem[subelem],tuple):
                    icount = icount + 1
                    f.write(str(icount)+"\n")
                    print(elem[subelem])
                    if icount == 1:
                        f.write("00:00:00,000 --> 0"+str(datetime.timedelta(seconds=elem[subelem][1]))[0:11].replace(".",",")+"\n")
                    else:
                        f.write("0"+str(datetime.timedelta(seconds=elem[subelem][0]))[0:11].replace(".",",")+" --> 0"+str(datetime.timedelta(seconds=elem[subelem][1]))[0:11].replace(".",",")+"\n")
                else:
                    f.write(elem[subelem]+"\n"+"\n")
except:
    pass

