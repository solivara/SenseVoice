# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re
from fastapi import FastAPI, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from typing_extensions import Annotated
from typing import List
from enum import Enum
import torchaudio
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO


class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"


model_dir = "/workspace/models/iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=os.getenv("SENSEVOICE_DEVICE", "cuda:0"))
m.eval()

regex = r"<\|.*\|>"

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """


@app.post("/api/v1/asr")
async def asr(
        files: Annotated[List[bytes], File(description="wav or mp3 audios in 16KHz")],
        keys: Annotated[str, Form(description="name of each audio joined with comma")],
        lang: Annotated[Language, Form(description="language of audio content")] = "auto"
):
    audios = []
    audio_fs = 0
    for file in files:
        file_io = BytesIO(file)
        data_or_path_or_list, audio_fs = torchaudio.load(file_io)
        data_or_path_or_list = data_or_path_or_list.mean(0)
        audios.append(data_or_path_or_list)
        file_io.close()
    if lang == "":
        lang = "auto"
    if keys == "":
        key = ["wav_file_tmp_name"]
    else:
        key = keys.split(",")
    res = m.inference(
        data_in=audios,
        language=lang,  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        ban_emo_unk=False,
        key=key,
        fs=audio_fs,
        **kwargs,
    )
    if len(res) == 0:
        return {"result": []}
    for it in res[0]:
        it["raw_text"] = it["text"]
        it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)
        it["text"] = rich_transcription_postprocess(it["text"])
    return {"result": res[0]}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
        file: Annotated[bytes, File(description="wav or mp3 audios in 16KHz")],
        model: Annotated[str, Form(description="model name")],
        language: Annotated[Language, Form(description="language of audio content")] = "auto"
):
    try:
        # Process audio file
        file_io = BytesIO(file)
        data_or_path_or_list, audio_fs = torchaudio.load(file_io)
        data_or_path_or_list = data_or_path_or_list.mean(0)
        file_io.close()

        # Run inference
        res = m.inference(
            data_in=[data_or_path_or_list],
            language=language,
            use_itn=False,
            ban_emo_unk=False,
            key=["audio_file"],
            fs=audio_fs,
            **kwargs,
        )

        if len(res) == 0 or len(res[0]) == 0:
            return {"text": ""}

        # Process result
        result = res[0][0]
        result["raw_text"] = result["text"]
        result["clean_text"] = re.sub(regex, "", result["text"], 0, re.MULTILINE)
        result["text"] = rich_transcription_postprocess(result["text"])

        return {"text": result["clean_text"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
