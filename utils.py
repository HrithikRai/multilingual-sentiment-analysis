import os
import random
from typing import Any, Dict, List, Union

import cohere
import numpy as np
import streamlit as st
import torch
from pyrate_limiter import Duration, Limiter, MemoryListBucket, RequestRate
from tqdm import tqdm

DEFAULT_TEXT = """Hey there, I am Hrithik Rai Saxena..."""

torchfy = lambda x: torch.as_tensor(x, dtype=torch.float32)

minute_rate = RequestRate(10_000, Duration.MINUTE)
limiter = Limiter(minute_rate, bucket_class=MemoryListBucket)
NUM_THREADS = 2 * os.cpu_count()  # 2 threads per cpu core is standard
N_MAX_RETRIES = 10


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_embeddings(co: cohere.Client,
                   texts: List[str],
                   model_name: str = 'multilingual-labse',
                   truncate: str = 'RIGHT',
                   batch_size: int = 2048) -> List[float]:

    @limiter.ratelimit("blobheart", delay=True)
    def get_embeddings_api(texts_batch: List[str]):

        for i in range(N_MAX_RETRIES):
            try:
                output = co.embed(model=model_name, texts=texts_batch, truncate=truncate)
                break
            except Exception as e:
                if i == (N_MAX_RETRIES - 1):
                    print(f"Exceeded max retries with error {e}")
                    raise f"Error {e}"
        return output.embeddings

    embeddings = []

    st_pbar = tqdm(range(0, len(texts), batch_size))
    for index in st_pbar:
        texts_batch = texts[index:index + batch_size]
        embeddings_batch = get_embeddings_api(texts_batch)  #list(pool.imap(get_embeddings_api, [texts_batch]))
        embeddings.append(embeddings_batch)
    return np.concatenate(embeddings, axis=0).tolist()

def streamlit_header_and_footer_setup():
    st.markdown(
        '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
        unsafe_allow_html=True)

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap');

        /* Background styling */
        body {
            background-color: #0d0d0d;
            color: #f0f0f0;
            font-family: 'Poppins', sans-serif;
        }

        /* Title container */
        .title-container {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            color: #f0f0f0;
            font-size: 2.5em;
            font-weight: bold;
            animation: fadeIn 3s ease-in-out;
        }

        /* Title Text Styling */
        .title-text {
            color: #917EF3;
            font-size: 3em;
            text-align: center;
            line-height: 1.2;
            background: linear-gradient(to right, #ff416c, #ff4b2b);
            -webkit-background-clip: text;
            color: transparent;
            animation: colorShift 5s infinite alternate-reverse ease-in-out;
        }

        /* Subtitle Styling */
        .subtitle-text {
            color: #a6a6a6;
            font-size: 1.2em;
            margin-top: 15px;
            text-align: center;
            animation: fadeIn 4s ease-in-out;
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes colorShift {
            0% { color: #ff4b2b; }
            50% { color: #ff416c; }
            100% { color: #917EF3; }
        }
        </style>

        <div class="title-container">
            <div>
                <div class="title-text">Multilingual Sentiment Analysis</div>
                <div class="subtitle-text">A Project by Hrithik Rai Saxena</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


    hide_st_style = """
        <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                content:'Cohere Inc';
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 2px;
            }
        </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

