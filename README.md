# hakol - speaker recognition

## chalenge
- [Video](https://rafaelcoil.sharepoint.com/sites/-2024_MSTEAMS/Shared%20Documents/General/Recordings/%D7%94%D7%9B%D7%9C%20%D7%91%D7%A7%D7%95%D7%9C%20WEBINAR-20240228_142941-Meeting%20Recording.mp4?web=1&referrer=Teams.TEAMS-ELECTRON&referrerScenario=MeetingChicletGetLink.view.view&isSPOFile=1)
- Thoughts:
  - Features: Gender/Language/Type of Noise
  - Seprate models
  - Clean signals
  - Finetune Nemo
  
 
  ![image](https://github.com/wzeyal/hakol/assets/64967130/0a12b4eb-00c1-46f6-aefd-8b272533adaf)

  Template Implemnation:
  [SpeakerRecognitionFromScratch](https://github.com/wq2012/SpeakerRecognitionFromScratch)
  

 
## udemy
- [Speaker Recognition Course Slides](https://drive.google.com/drive/folders/1BDuu5gkTSDaLtYPHMUM7pVIyfogozVj_?usp=sharing)
- 
 
## papers with code
- [Speaker Recognition](https://paperswithcode.com/task/speaker-recognition)

## Frameworks
### Nvidia Nemo
>NVIDIA NeMo™ is an end-to-end, cloud-native framework to build, customize, and deploy generative AI models anywhere. It includes training and inferencing frameworks, guardrailing toolkits, data curation tools, and pretrained models, offering enterprises an easy, cost-effective, and fast way to adopt generative AI.
- [Speaker Recognition](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/intro.html)
- [Notebooks](https://github.com/NVIDIA/NeMo/tree/main/tutorials/speaker_tasks)
- [InferenceGitHub](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/speakerverification_speakernet)

```
!pip install Cython
!pip install nemo_toolkit['all']

import nemo.collections.asr as nemo_asr
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_small')
decision = speaker_model.verify_speakers('manners.wav','king.wav')
```

### HuggingFace
- [speaker_ver](https://huggingface.co/models?search=speaker_ver)
- [Finetune example](https://huggingface.co/pgwi/en_tr_titanet_large/tree/main)
  
### pyannote
>pyannote.audio is an open-source toolkit written in Python for speaker diarization. Based on PyTorch machine learning framework, it comes with state-of-the-art pretrained models and pipelines, ?that can be further finetuned to your own data for even better performance.
- [Github](https://github.com/pyannote/pyannote-audio)
### noise remove
- [remove-noise-from-audio](https://medium.com/@devesh_kumar/how-to-remove-noise-from-audio-in-less-than-10-seconds-8a1b31a5143a)

## Datasets
- [OpenSlr](https://openslr.org/index.html)
- [commonvoice](https://commonvoice.mozilla.org/en/datasets)
- [Moviesoundclips.Net](http://www.moviesoundclips.net/)

## Evaluation
- TBD

## Utilites
- [SoX](https://sourceforge.net/projects/sox)
  > SoX is the Swiss Army Knife of sound processing utilities. It can convert audio files to other popular audio file types and also apply sound effects and filters during the conversion.
- [Audacity](https://www.audacityteam.org/)
  > Audacity is the world's most popular audio editing and recording app
