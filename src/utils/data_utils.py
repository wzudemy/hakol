import json
import logging
import math
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor

import torch
from tqdm import tqdm

import numpy as np
import pandas as pd
import src.config as C
from src.utils.audio_utils import convert_to_nemo_format_using_pydub, detect_gender
from src.utils.file_utils import group_file_by_speaker, create_stub_folder, get_base_filenames, init_logging
from glob import glob
from sklearn.model_selection import train_test_split

# reproducibility
seed = 41
random.seed(seed)
np.random.seed(seed)
cosine_sim = torch.nn.CosineSimilarity(dim=-1)

# audio_files_path = 'speakathon_data_subset/challenge'
hack_group_name = "group_4"

logger = logging.getLogger(__name__)


def preprocess_data():
    filelist_csv = C.SPEAKATHON_DATA_SUBSET / 'hackathon_train.csv'
    filtered_filelist_csv = C.SPEAKATHON_DATA_SUBSET / 'hackathon_train_filtered.csv'
    src_folder = C.SPEAKATHON_DATA_SUBSET / 'wav_files'
    dest_folder = C.SPEAKATHON_DATA_SUBSET / 'nemo'

    if C.STUB:
        generate_stub_dataset(src_folder, 100)

        filelist_csv = C.DATA_DIR / 'stub' / 'stub.csv'
        filtered_filelist_csv = C.DATA_DIR / 'stub' / 'filtered_filelist.csv'
        src_folder = C.DATA_DIR / 'stub' / 'wav_files_subset'
        dest_folder = C.DATA_DIR / 'stub' / 'nemo'

    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)

    # filter the data for
    df = pd.read_csv(filelist_csv)
    filtered_df = df.groupby(['speaker']).head(C.MAX_SPEAKERS_COUNT_FILTER)
    filtered_df.to_csv(filtered_filelist_csv)

    group_file_by_speaker(filtered_filelist_csv, src_folder, dest_folder)

    files_list = glob(os.path.join(dest_folder, "**", "*.wav"), recursive=True)

    logger.info('convert_to_nemo_format_using_pydub start')
    with ProcessPoolExecutor() as pool:
        nemo_wav_files = pool.map(convert_to_nemo_format_using_pydub, files_list)
    logger.info('convert_to_nemo_format_using_pydub end')

    print(len(list(nemo_wav_files)))

    return dest_folder


def read_csv_to_dict(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)

    df['file'] = df['file'].apply(lambda x: x.split('.')[0])
    # Set the 'file' column as the index
    df.set_index('file', inplace=True)

    # Convert the DataFrame to a dictionary
    data_dict = df.to_dict(orient='index')

    return data_dict


def get_wav_dict(key, wav_mapping):
    orig_key = os.path.basename(key)
    orig_key = orig_key.split("_")[0]
    if orig_key in wav_mapping:
        return wav_mapping.get(orig_key)
    else:
        print("error")
        return None


def get_speaker_lang(df, speaker):
    return df[df.speaker == speaker]['language'].iloc[0]


def get_utterance_lang(df, file):
    return df[df.file == file]['language'].iloc[0]


def get_utterance_noise_type(df, file):
    return df[df.file == file]['noise_type'].iloc[0]


def get_spk_to_utt(df, audio_files_path):
    # Creta a dictionary of speaker to utterances
    spk_to_utts = dict()

    for index, row in df.iterrows():
        file_path = row['file']
        spk = row['speaker']
        file_path = os.path.join(audio_files_path, file_path)
        if not os.path.exists(file_path):
            print(f"Found invalid file: {file_path}")

        if spk not in spk_to_utts:
            spk_to_utts[spk] = [file_path]
        else:
            spk_to_utts[spk].append(file_path)
    return spk_to_utts


def generate_stub_dataset(src_folder, dst_folder, num_duplicates):
    logger.info('start')
    create_stub_folder(src_folder, dst_folder, num_duplicates)
    filenames = glob(os.path.join(dst_folder, "**", "*.wav"), recursive=True)

    wav_mapping = read_csv_to_dict(
        'speakathon_data_subset/hackathon_train_subset.csv')

    # Create a list starting from 1000 and ending at 1999
    # speakers = list(range(start_value, start_value + num_items))
    # Create a list repeating the given numbers
    # speakers_ids = [speakers[i % len(speakers)] for i in range(len(filenames))]
    stub_df = pd.DataFrame()
    stub_df['file'] = [os.path.basename(filename) for filename in filenames]

    wave_mapping_full_dict = {}
    for filename in filenames:
        wave_mapping_full_dict[filename] = get_wav_dict(filename, wav_mapping)

    list_of_1000_items = list(range(1000, 1999))
    num_rows = len(stub_df)
    stub_df['speaker'] = [random.choice(list_of_1000_items) for _ in range(num_rows)]
    stub_df['language'] = [wave_mapping_full_dict[filename]['language'] for filename in filenames]
    stub_df['noise_type'] = [wave_mapping_full_dict[filename]['noise_type'] for filename in filenames]
    stub_df.to_csv(os.path.join('speakathon_data_subset', 'hackathon_train.csv'), index=False)
    logger.info('end')


def choose_anchor_utt(speaker, spk_to_utts):
    """
    Selects an anchor utterance and another random utterance for a given speaker.

    Parameters:
    - speaker: The identifier for the speaker of interest.
    - spk_to_utts: A dictionary mapping each speaker to a list of their utterances (file paths).

    Returns:
    - A tuple containing:
        - speaker: The identifier of the speaker.
        - anchor_audio_basename: The base name of the anchor audio file (without folder path).
        - same_speaker_utt_basename: The base name of another utterance from the same speaker.
    """
    anchor_audio = random.choice(spk_to_utts[speaker])

    # get another utterance from the same speaker
    all_speaker_utt = spk_to_utts[speaker]

    all_speaker_utt.remove(anchor_audio)

    random.shuffle(all_speaker_utt)
    same_speaker_utt = all_speaker_utt[0]

    return speaker, os.path.basename(anchor_audio), os.path.basename(same_speaker_utt)


def create_group(df, anchor_speaker, spk_to_utts):
    """
    Creates a group of audio files based on a given anchor speaker. The group is formed by selecting
    an anchor utterance from the anchor speaker and additional utterances from different speakers
    that share the same language as the anchor.

    Parameters:
    - df: DataFrame containing metadata about the audio files, including speaker, file path, language, and noise type.
    - anchor_speaker: The identifier for the anchor speaker.
    - spk_to_utts: A dictionary mapping each speaker to their utterances (audio file paths).

    Returns:
    - A tuple containing information about the created group, including:
        - group_audio_files: A list of the base names of the audio files in the group.
        - group_audio_speaker: A list of speaker identifiers corresponding to each file in the group.
        - anchor_speaker: The identifier of the anchor speaker.
        - anchor_audio: The base name of the anchor audio file.
        - anchor_type: The noise type of the anchor audio file.
        - same_speaker_utt: The base name of another utterance from the anchor speaker.
        - group_audio_type: A list of noise types corresponding to each audio file in the group.

    The function first selects an anchor utterance for the anchor speaker and another utterance from the
    same speaker. It then identifies additional speakers who speak the same language as the anchor speaker
    and randomly selects utterances from these speakers to form the group.
    """

    speakers = list(spk_to_utts.keys())

    # Choose an anchor audio file for the anchor speaker
    anchor_speaker, anchor_audio, same_speaker_utt = choose_anchor_utt(anchor_speaker, spk_to_utts)

    # Get the anchor's language
    anchor_lang = get_speaker_lang(df, anchor_speaker)
    anchor_type = get_utterance_noise_type(df, anchor_audio)

    # Remove anchor speaker temporarily
    speakers.remove(anchor_speaker)

    group_audio_files = []
    group_audio_speaker = []
    group_audio_type = []

    # Randomly decide the number of files in the group
    num_files_in_group = random.randint(5, 20)

    group_audio_files.append(same_speaker_utt)
    group_audio_speaker.append(anchor_speaker)
    group_audio_type.append(get_utterance_noise_type(df, same_speaker_utt))

    num_files_in_group -= 1  # Decrement since we added one from the anchor speaker

    # Randomly select speakers from the same language
    curr_speakers = set(df[(df.language == anchor_lang) & (df.speaker != anchor_speaker)]['speaker'].values.tolist())

    selected_speakers = random.sample(list(curr_speakers), min(num_files_in_group, len(curr_speakers)))

    for speaker in selected_speakers:
        group_utt = random.choice(spk_to_utts[speaker])
        group_audio_files.append(group_utt)
        group_audio_speaker.append(speaker)
        group_audio_type.append(get_utterance_noise_type(df, os.path.basename(group_utt)))

    group_audio_files = [os.path.basename(f) for f in group_audio_files]
    combined_lists = list(zip(group_audio_files, group_audio_speaker))
    random.shuffle(combined_lists)
    group_audio_files, group_audio_speaker = zip(*combined_lists)
    return group_audio_files, group_audio_speaker, anchor_speaker, anchor_audio, anchor_type, same_speaker_utt, group_audio_type


def create_dataset(df, spk_to_utts, num_groups, speakers):
    """
    Generates dataset of groups where each group contains an anchor audio file
    and additional audio files.

    Parameters:
    - df: DataFrame containing audio file metadata, including speaker IDs, file paths, languages, and noise types.
    - spk_to_utts: Dictionary mapping speaker IDs to their utterances (list of audio file paths).
    - num_groups: The desired number of groups to create in the dataset.

    Returns:
    - groups_df: A pandas DataFrame with columns detailing each group's composition, including the group ID, anchor file,
                 anchor speaker, group files, speakers for each file in the group, and the target label (file from the
                 same speaker as the anchor).
    """

    dataset = []

    # Filter speakers to only those with at least two audio files
    speakers_with_at_least_two_utts = [speaker for speaker in speakers if len(spk_to_utts[speaker]) >= 2]

    # Ensure there are enough speakers to meet the num_groups requirement
    if len(speakers_with_at_least_two_utts) >= num_groups:
        # Directly sample from the filtered list of speakers
        anchor_speakers = random.sample(speakers_with_at_least_two_utts, num_groups)
    else:
        raise ValueError(f"Not enough speakers with at least two utterances to form {num_groups} groups.")

    for i in range(num_groups):
        anchor = anchor_speakers[i]
        # create the group for this anchor
        group, group_audio_speaker, anchor_speaker_id, anchor_utt, anchor_type, label, group_audio_type = create_group(
            df, anchor, spk_to_utts)

        dataset.append({
            'group_index': i,
            'group': group,
            'group_audio_speaker': group_audio_speaker,
            'group_audio_type': group_audio_type,
            'anchor_speaker': anchor_speaker_id,
            'anchor': anchor_utt,
            'anchor_type': anchor_type,
            'label': label
        })

    # create a new dataframe with columns: group_id, anchor_file, group_file, group_label, group_audio_type
    rows = []
    for group in dataset:
        group_index = group['group_index']
        anchor_file = os.path.basename(group['anchor'])
        anchor_speaker = group['anchor_speaker']
        anchor_type = group['anchor_type']
        group_label = os.path.basename(group['label'])

        for i, f in enumerate(group['group']):
            group_file = os.path.basename(f)
            group_audio_speaker = group['group_audio_speaker'][i]
            group_audio_type = group['group_audio_type'][i]
            row = {'group_id': group_index, 'group_audio_speaker': group_audio_speaker, 'group_label': group_label,
                   'anchor_file': anchor_file, 'anchor_speaker': anchor_speaker, 'anchor_type': anchor_type,
                   'group_file': group_file, 'group_audio_type': group_audio_type}
            rows.append(row)

    groups_df = pd.DataFrame(rows)

    # Reorder the columns:
    # group_id:            The ID of the group
    # anchor_file:         The file which is the anchor
    # anchor_speaker:      The speaker id of the person speaking in the anchor_file
    # group_file:          A file in the group
    # group_audio_speaker: The speaker is that is speaking in the group_file
    # group_audio_type:    The noise type of this group file
    # group_label:         The label we want to predict, the name of the group_file in which the speaker is the same as the anchor_file

    groups_df = groups_df[
        ["group_id", "anchor_file", "anchor_speaker", "anchor_type", "group_file", "group_audio_speaker",
         "group_audio_type", "group_label"]]

    return groups_df


def create_validation_dataset(train_csv_path, audio_files_path):
    validation_path = C.DATA_DIR / 'validation.csv'
    train_df = pd.read_csv(train_csv_path)

    unique_speakers = train_df['speaker'].unique()

    # Split the unique speakers into train and validation sets
    train_speakers, validation_speakers = train_test_split(unique_speakers, test_size=0.2, random_state=seed)

    # Filter dataframe based on the split speakers
    train_df_new = train_df[train_df['speaker'].isin(train_speakers)]
    validation_df = train_df[train_df['speaker'].isin(validation_speakers)]

    print(train_df.shape, train_df_new.shape, validation_df.shape)

    spk_to_utts = get_spk_to_utt(validation_df, audio_files_path)

    speakers = set(spk_to_utts.keys())

    NUM_GROUPS_TO_CREATE = min(len(speakers), 1000)

    groups_df = create_dataset(validation_df, spk_to_utts, NUM_GROUPS_TO_CREATE, speakers)

    groups_df.to_csv(validation_path, index=False)

    return groups_df


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


def get_embeddings(anchor_file, group_utterances, encoder, audio_files_path):
    """
    Generates embeddings for an anchor file and a group of utterances to compare against the anchor.

    This function prepares a batch for inference by first adding the anchor file to the beginning
    of the list of group utterances. It then constructs full paths for these files, runs the
    inference to get embeddings, and separates the embeddings of the anchor from the group.

    Parameters:
    - anchor_file (str): The file path of the anchor audio file.
    - group_utterances (list): A list of file paths for the group's utterances.

    Returns:
    - tuple: A tuple containing three elements:
        - anchor_emb (np.array): The embedding of the anchor file.
        - group_emb (list): A list of embeddings for the group utterances.
        - group_utterances (list): The original list of group utterances file paths,
          excluding the anchor file.
    """

    # Insert anchor_file as the first element for batch inference
    group_utterances.insert(0, anchor_file)

    # Construct full paths for each file in the group_utterances
    group_utterances_full = [os.path.join(audio_files_path, f) for f in group_utterances]

    # Run inference to get embeddings for the anchor and the group utterances
    all_emb = run_inference(group_utterances_full, encoder)

    # Separate the embeddings of the anchor from the group
    anchor_emb = all_emb[0]  # Embedding of the anchor file
    group_emb = all_emb[1:]  # Embeddings of the group utterances

    # Restore the original group_utterances list by removing the anchor file
    group_utterances = group_utterances[1:]

    return anchor_emb, group_emb, group_utterances


def get_closest_speaker(anchor_emb, group_emb, group_utterances):
    """
    Identifies the utterance from a group that is closest to the anchor utterance based on cosine similarity.

    Parameters:
    - anchor_emb (torch.Tensor): The embedding vector of the anchor utterance.
    - group_emb (torch.Tensor):  A tensor of embedding vectors for the group utterances.
    - group_utterances (list):   A list of file paths for each utterance in the group.

    Returns:
    - str: The file path of the group utterance most similar to the anchor utterance.
    """
    distances = []
    # Compute cosine similarity between the anchor and each group utterance
    for i in range(group_emb.shape[0]):
        similarity = cosine_sim(anchor_emb, group_emb[i])  # Compute similarity
        distances.append(similarity.item())  # Convert tensor to Python scalar and append

    # Identify the index of the highest similarity score
    max_index = np.argmax(distances)

    # Retrieve the corresponding group utterance identifier
    same_speaker_utt = group_utterances[max_index]

    return same_speaker_utt


def submit_challenge_results(group_name, same_speaker_utt_lst):
    """
    Saves a list of selected utterances (one per group) into a CSV file that can be submitted for scoring.

    Parameters:
    - same_speaker_utt_lst (list): A list of utterances (files) identified as from the same speaker as the anchor in their group.
      Utterance in index k corresponds to the selected utterance of group k
    """
    # Convert the list to a DataFrame
    df = pd.DataFrame(same_speaker_utt_lst, columns=["same_speaker_utt"])

    # Write to CSV
    df.to_csv(f"{group_name}_results.csv", index=False)

def sign(float_number):
    return float_number > 0


def generate_results(encoder, challenge_path, audio_files_path):
    # challenge_path = 'speakathon_data_subset/groups_challenge_validation.csv'
    groups_challenge_df = pd.read_csv(challenge_path)
    num_groups = len(groups_challenge_df['group_id'].unique())

    if C.HP_FILTER_PREDICTED_SAME_GENDER:
        file_path = '/workdir/data/gender_prob.json'
        with open(file_path, 'r') as json_file:
            gender_prob = json.load(json_file)


    same_speaker_utt_lst = []

    logger.info("Iterate through each group in the dataset")
    # Iterate through each group in the dataset
    for group_id in tqdm(range(num_groups)):
        # Extract the dataframe for the current group
        group_df = groups_challenge_df[groups_challenge_df.group_id == group_id]

        # Retrieve the anchor file for the current group
        anchor_file = group_df['anchor_file'].iloc[0]
        
        # Collect all utterance files associated with the current group
        group_utterances = group_df['group_file'].values.tolist()

        # TODO: consider gender
        if C.HP_FILTER_PREDICTED_SAME_GENDER:
            anchor_gender = gender_prob.get(anchor_file, 0)
            group_genders = [gender_prob.get(group_utterance, 0) for group_utterance in group_utterances]
            same_gender_list = [item for item, gender in zip(group_utterances, group_genders) if sign(anchor_gender)==sign(gender)]
            if len(same_gender_list) > 0:
                group_utterances = same_gender_list

        # Obtain embeddings for the anchor and group utterances
        anchor_emb, group_emb, group_utterances = get_embeddings(anchor_file, group_utterances, encoder, audio_files_path)

        # Determine the group utterance most similar to the anchor in embedding space
        same_speaker_utt = get_closest_speaker(anchor_emb, group_emb, group_utterances)

        # Append the identified utterance to the list of same speaker utterances
        same_speaker_utt_lst.append(same_speaker_utt)
    return same_speaker_utt_lst

    # After processing all groups, submit the results
    


def calculate_score_on_validation():
    GROUP_CHALLENGE_VALIDATION = C.DATA_DIR / 'validation.csv'
    RESULTS_FILE = 'group_4_results.csv'
    groups_df = pd.read_csv(GROUP_CHALLENGE_VALIDATION)
    NUM_GROUPS = len(groups_df['group_id'].unique())
    res_df = pd.read_csv(RESULTS_FILE)
    correct = 0

    for group_id in range(NUM_GROUPS):
        group_df = groups_df[groups_df.group_id == group_id]
        true_label = group_df['group_label'].iloc[0]

        predicted_label = res_df.iloc[group_id][0]
        if true_label == predicted_label:
            correct += 1

    challenge_score = correct / NUM_GROUPS

    print(f"validation challenge score: {challenge_score:.2%}%")
    return challenge_score * 100

def convert_wave_to_nemo():
    src_folder = C.SPEAKATHON_DATA_SUBSET / 'wav_files'
    files_list = glob(os.path.join(src_folder, "**", "*.wav"), recursive=True)
    logger.info('convert_wave_to_nemo start')
    with ProcessPoolExecutor() as pool:
        nemo_wav_files = pool.map(convert_to_nemo_format_using_pydub, files_list)
    logger.info('convert_wave_to_nemo end')

if __name__ == "__main__":
    init_logging()

    file_path1 = '/workdir/data/gender_prob.json'
    with open(file_path1, 'r') as json_file:
        gender_prob= json.load(json_file)

    file_path2 = '/workdir/data/gender_dict.json'
    with open(file_path2, 'r') as json_file:
        gender_dict = json.load(json_file)

    gender_prob2 = {key: -1.0 if value=="female" else 1.0 for key, value in gender_dict.items()}

    gender_prob.update(gender_prob2)
    with open('/workdir/data/gender_prob2.json', "w") as json_file:
        json.dump(gender_prob, json_file)


    print(sign(0.05))
    print(sign(-0.001))
    is_generate_stub_dataset = False
    is_convert_wave_to_nemo = True
    is_create_validation_dataset = False
    is_generate_results = False
    is_calculate_score_on_validation = False

    # TODO:
    # replace encoder with our own
    import nemo.collections.asr as nemo_asr

    # speaker_model_path = C.DATA_DIR / f"{C.NEMO_MODEL_NAME}_ft.nemo"
    speaker_model_path = C.DATA_DIR / 'titanet-large_not_segment_ft.nemo'
    encoder = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(speaker_model_path)
    audio_files_path =  C.DATA_DIR / 'wav_files'
    challenge_path = C.DATA_DIR / 'groups_challenge_validation.csv'
    hackathon_train_path = C.DATA_DIR / 'hackathon_train.csv'
    validation_path = C.DATA_DIR / 'validation.csv'

    if is_generate_stub_dataset:
        logger.info("is_generate_stub_dataset")
        src_folder: str = 'speakathon_data_subset/wav_files_subset'
        dest_folder: str = 'speakathon_data_subset/challenge'

        if os.path.exists(dest_folder):
            shutil.rmtree(dest_folder)

        generate_stub_dataset(src_folder, dest_folder, 600)

    if is_convert_wave_to_nemo:
        convert_wave_to_nemo()

    if is_create_validation_dataset:
        logger.info("is_create_validation_dataset")
        create_validation_dataset(hackathon_train_path, audio_files_path)

    if is_generate_results:
        logger.info("is_generate_results")
        generate_results(encoder, validation_path, audio_files_path)

    if is_calculate_score_on_validation:
        logger.info("calculate_score_on_validation")
        calculate_score_on_validation()
