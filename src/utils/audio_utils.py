import itertools
import os
import soundfile as sf
import librosa
import torch
from pydub import AudioSegment
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

from utils.file_utils import check_extension, get_files_in_folder
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelw = Wav2Vec2ForSequenceClassification.from_pretrained(
    'alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech')
processor = Wav2Vec2FeatureExtractor.from_pretrained(
    'alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech')


def process_list_in_chunks(lst, process_chunk, chunk_size=4):
    chunks = [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
    # list_of_lists =  list(map(process_chunk, chunks))
    # res = list(itertools.chain.from_iterable(list_of_lists))
    result = []
    for chunk in tqdm(chunks, desc="Processing chunks"):
        result.extend(process_chunk(chunk))
    return result



def infrennce_gender(input_values):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # input_values.to(device)
        # modelw.to(device)
        result = modelw(input_values).logits
        prob = torch.nn.functional.softmax(result, dim=1)

    torch.cuda.empty_cache()
    return torch.argmax(prob, dim=1).detach().cpu().numpy()

def detect_gender(wave_folder):
    import torch
    import numpy as np
    from scipy.io.wavfile import read

    wave_files = get_files_in_folder(wave_folder)

    wave_files = wave_files[:8]

    sound_array = []
    for wave_file in wave_files:
        output = read(wave_file)
        wave_file = np.array(output[1], dtype=float)
        sound_array.append(wave_file)

    # Determine the maximum length of the sublists
    max_length = max(len(sublist) for sublist in sound_array)

    # Create an empty array to hold the padded sublists
    padded_array = np.zeros((len(sound_array), max_length))

    # Pad each sublist to match the length of the longest sublist
    for i, sublist in enumerate(sound_array):
        padded_array[i, :len(sublist)] = sublist

    input_values = processor(sound_array, sampling_rate=16000, padding='longest', return_tensors='pt').input_values

    # result = modelw(input_values).logits

    # result = infrennce_gender(input_values)

    binary_list = process_list_in_chunks(input_values, infrennce_gender, )
    gender_list = [modelw.config.id2label[x] for x in binary_list]

    return gender_list

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def convert_to_nemo_format_using_pydub(
        input_file: str,
        target_sr: int = 16_000,
        outfile: str = None,
        target_dBFS: float = -20.0,
):
    _, file_extension = os.path.splitext(input_file)
    if file_extension == '.mp3':
        input_audio = AudioSegment.from_mp3(input_file)
    elif file_extension == '.wav':
        input_audio = AudioSegment.from_wav(input_file)
    else:
        raise ValueError(f"Unsupported extension: {file_extension}")

    output_audio = input_audio.set_channels(1)
    output_audio = output_audio.set_frame_rate(target_sr)

    # Apply AGC (Automatic Gain Control)
    output_audio = match_target_amplitude(output_audio, target_dBFS=target_dBFS)

    if outfile:
        output_audio.export(f"{outfile}", format="wav")
    else:
        outfile = os.path.splitext(input_file)[0]
        outfile = f"{outfile}_nemo.wav"
        output_audio.export(outfile, format="wav")
    return outfile



if __name__ == "__main__":
    test_folder = '/home/eyalshw/github/wzudemy/hakol/data/subset/wav_files_subset'

    classification = detect_gender(test_folder)
    print(classification)
