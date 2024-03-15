import os

import torch
from nemo.collections.asr.parts.utils.speaker_utils import embedding_normalize
from tqdm import  tqdm
try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield
import numpy as np
import json
import pickle as pkl

from omegaconf import OmegaConf
from tqdm import tqdm




@torch.no_grad()
def calc_score(embs1, embs2):
    # Length Normalize
    X = embs1 / torch.linalg.norm(embs1)
    Y = embs2 / torch.linalg.norm(embs2)
    # Score
    similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
    similarity_score = (similarity_score + 1) / 2
    return similarity_score

@torch.no_grad()
def verify_speakers(spk_model, anchor, group_file_list):
    """
    Verify if two audio files are from the same speaker or not.

    Args:
        path2audio_file1: path to audio wav file of speaker 1
        path2audio_file2: path to audio wav file of speaker 2

    Returns:
        True if both audio files are from same speaker, False otherwise
    """
    anchor_embs = spk_model.get_embedding(anchor).squeeze()
    scores = [
        calc_score(anchor_embs, spk_model.get_embedding(cand).squeeze())
        for cand in group_file_list
    ]
    return torch.stack(scores).argmax().item()