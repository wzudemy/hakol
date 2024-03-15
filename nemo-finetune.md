# Model

- Finetuning using nvidia nemo finetuining
- Based on:
  -   [NVIDIA speaker task tutorial](https://github.com/NVIDIA/NeMo/blob/main/tutorials/speaker_tasks/Speaker_Identification_Verification.ipynb)
  -   [reviesd version](https://github.com/wzudemy/hakol-1/blob/main/notebooks/Speaker_Identification_Verification.ipynb)


# Dataset
- Chanllenge Subset

# Output
- speaker moedel encoder in nemo format

# Validation
- Using the [Speakathon](https://github.com/wzudemy/hakol/blob/eyals/notebooks/Speakathon.ipynb) notebook
- Change the run_inference function to:
```
def run_inference(utt_list, encoder):
    emb_arr = []
    for utt in utt_list:
        # signal, sr = torchaudio.load(utt)
        # signal = torchaudio.functional.resample(signal, orig_freq=sr, new_freq=TARGET_SR)
        # signal = signal.to(DEVICE)
        # signal_emb = encoder(signal)
        signal_emb = encoder.get_embedding(utt).squeeze()
        emb_arr.append(signal_emb)
    return torch.stack(emb_arr).squeeze(1)
```

# In the hackton
- finetune on challenge
- Imporvements:
  - finetune (before/after) on cv
  - Speakathon notebook::get_closest_speaker: compare only same gender
  - Add noise removal
  - Split models by the audio type (comm/noise/clean)


  
- 
