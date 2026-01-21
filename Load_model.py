import numpy as np
from efficient_data_loader import get_dataloader

from transformers import AutoProcessor, Data2VecAudioModel, AutoModel, AutoFeatureExtractor
import torch
import os
import librosa

def load_train_data_loader(train_path, batch_size, sample_len, return_recording=False):
    train_loader = get_dataloader(recording_path=train_path, sample_rate=16000,
                                  batch_size=int(batch_size), sample_len_sec=sample_len,
                                  return_recording=return_recording, shuffled=False)
    return train_loader

def load_test_data_loader(test_path, batch_size, sample_len, return_recording=False):

    test_loader = get_dataloader(recording_path=test_path, sample_rate=16000,
                                  batch_size=int(batch_size), sample_len_sec=sample_len, return_recording=return_recording)
    return test_loader

def MelSpec(test_dir):
    results = []
    labels = []
    recordings = []
    timestamps = []
    # for batch, label in test_dir:
    for batch, label, recording, start_time in test_dir:
        for waveform in batch:
            mel_spectrogram = librosa.feature.melspectrogram(y=waveform.numpy(), sr=16000, n_fft=1024, hop_length=512, n_mels=128, fmax=8000)
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram).mean(axis=1)
            results.append(log_mel_spectrogram)
        labels.extend(label)
        recordings.extend(list(recording))
        timestamps.extend(start_time.cpu().detach().numpy())

    return np.array(results), np.array(labels), recordings, timestamps

def prediction_HuBERT(test_dir):
    from transformers import HubertModel
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    results = []
    labels = []
    recordings = []
    timestamps = []
    for batch, label, recording, start_time in test_dir:
        # Forward pass through the model to extract features
        with torch.no_grad():
            output = model(batch)
            result = output.last_hidden_state.mean(dim=1)
        # results.extend(torch.flatten(result['extract_features'], start_dim=1).detach().numpy())
        results.extend(result.cpu().detach().numpy())
        labels.extend(label.cpu().detach().numpy())
        recordings.extend(list(recording))
        timestamps.extend(start_time.cpu().detach().numpy())
    result = np.array(results)
    labels = np.array(labels)
    return result, labels, recordings, timestamps


def prediction_AudioMAE(test_dir, mean_pooling=True):
    import tempfile
    import soundfile as sf

    model = AutoModel.from_pretrained("hance-ai/audiomae", trust_remote_code=True)
    model.eval()

    results, labels = [], []
    recordings = []
    timestamps = []

    for batch, label, recording, start_time in test_dir:   # batch: waveform tensor [B, T]
        for i in range(batch.size(0)):
            waveform = batch[i].cpu().numpy()

            # Write waveform to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                sf.write(tmp.name, waveform, 16000)  # 16kHz expected

                # Forward pass with the model (takes file path)
                with torch.no_grad():
                    emb = model(tmp.name)   # (latent_dim, freq_dim, time_dim)
                    if mean_pooling:
                        pooled = emb.reshape(768, -1).transpose(0, 1)  # shape: (8*64, 768)
                    else:
                        pooled = emb

            results.append(pooled.cpu().numpy())
            labels.append(label[i].cpu().numpy())
            recordings.extend(list(recording))
            timestamps.extend(start_time.cpu().detach().numpy())

    return np.array(results), np.array(labels), recordings, timestamps

def prediction_WavLM(test_dir):
    from transformers import WavLMModel, Wav2Vec2FeatureExtractor

    # Load the feature extractor (no tokenizer needed)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")

    results = []
    labels = []
    recordings = []
    timestamps = []
    for batch, label, recording, start_time in test_dir:
        inputs = feature_extractor(batch.detach().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            # last_hidden_state: (batch_size, sequence_length, hidden_size)
            # hidden_states: all transformer layer outputs
            hidden_states = outputs.last_hidden_state
            # Mean pooling across time
            result = hidden_states.mean(dim=1)  # shape: (batch, hidden_size)
        # results.extend(torch.flatten(result['extract_features'], start_dim=1).detach().numpy())
        results.extend(result.cpu().detach().numpy())
        labels.extend(label.cpu().detach().numpy())
        recordings.extend(list(recording))
        timestamps.extend(start_time.cpu().detach().numpy())
    # features: Tensor with shape [num_frames, feature_dim]
    result = np.array(results)
    labels = np.array(labels)
    return result, labels, recordings, timestamps

def prediction_BEATS(test_dir, mean_pooling = True):
    from speechbrain.lobes.models.beats import BEATs

    BEATs_model = BEATs(
        ckp_path="models/BEATs_iter3_plus_AS2M.pt",
        freeze=True
    )

    results = []
    labels = []
    recordings = []
    timestamps = []
    for batch, label, recording, start_time in test_dir:
        durations = torch.tensor([10.0]*batch.shape[0])
        output = BEATs_model.extract_features(batch, durations)[0]
        if mean_pooling:
            result = output.mean(dim=1)
        else:
            result = output
        # padding_mask = torch.zeros(1, 10000).bool()
        # result = BEATs_model.extract_features(batch, padding_mask=padding_mask)[0]
        print(result.shape)
        results.extend(result.cpu().detach().numpy())
        labels.extend(label.cpu().detach().numpy())
        recordings.extend(list(recording))
        timestamps.extend(start_time.cpu().detach().numpy())
    # features: Tensor with shape [num_frames, feature_dim]
    result = np.array(results)
    labels = np.array(labels)
    return result, labels, recordings, timestamps

def prediction_HuBERTAS(test_dir):

    # Load processor + model
    processor = AutoFeatureExtractor.from_pretrained("ALM/hubert-base-audioset")
    model = AutoModel.from_pretrained("ALM/hubert-base-audioset")
    model.eval()  # put in inference mode
    results = []
    labels = []
    recordings = []
    timestamps = []
    for batch, label, recording, start_time in test_dir:
        inputs = processor(batch.squeeze().numpy(),
                           sampling_rate=16000,
                           return_tensors="pt",
                           padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        result = hidden_states.mean(dim=1)
        results.extend(result.cpu().detach().numpy())
        labels.extend(label.cpu().detach().numpy())
        recordings.extend(list(recording))
        timestamps.extend(start_time.cpu().detach().numpy())
        # features: Tensor with shape [num_frames, feature_dim]
    result = np.array(results)
    labels = np.array(labels)
    return result, labels, recordings, timestamps

def prediction_wav2vec(test_dir):
    from transformers import Wav2Vec2Processor, Wav2Vec2Model

    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()  # put in inference mode
    results = []
    labels = []
    recordings = []
    timestamps = []
    for batch, label, recording, start_time in test_dir:
        inputs = processor(batch.squeeze().numpy(),
                           sampling_rate=16000,
                           return_tensors="pt",
                           padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        result = hidden_states.mean(dim=1)
        print(result.shape)
        results.extend(result.cpu().detach().numpy())
        labels.extend(label.cpu().detach().numpy())
        recordings.extend(list(recording))
        timestamps.extend(start_time.cpu().detach().numpy())
        # features: Tensor with shape [num_frames, feature_dim]
    result = np.array(results)
    labels = np.array(labels)
    return result, labels, recordings, timestamps


def prediction_BirdMAE(test_dir):
    from transformers import AutoFeatureExtractor, AutoModel
    MODEL_NAME = "DBD-research-group/Bird-MAE-Base"

    # import model, feature extractor, tokenizer
    processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()  # put in inference mode
    results = []
    labels = []
    recordings = []
    timestamps = []
    for batch, label, recording, start_time in test_dir:
        inputs = processor(batch.squeeze().numpy(),
                           # sampling_rate=16000,
                           return_tensors="pt",
                           padding=True)

        with torch.no_grad():
            outputs = model(inputs)
            result = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        # result = hidden_states.mean(dim=1)
        print(result.shape)
        results.extend(result.cpu().detach().numpy())
        labels.extend(label.cpu().detach().numpy())
        recordings.extend(list(recording))
        timestamps.extend(start_time.cpu().detach().numpy())
        # features: Tensor with shape [num_frames, feature_dim]
    result = np.array(results)
    labels = np.array(labels)
    return result, labels, recordings, timestamps

def prediction_Perch2(test_dir):
    import tensorflow as tf
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    # Load the model.
    model = tf.saved_model.load('models/perch20_cpu')
    results = []
    labels = []
    recordings = []
    timestamps = []
    for batch, label, recording, start_time in test_dir:
        model_outputs = model.signatures['serving_default'](inputs=batch)
        result = model_outputs['embedding']
        results.extend(result.numpy())
        labels.extend(label.cpu().detach().numpy())
        recordings.extend(list(recording))
        timestamps.extend(start_time.cpu().detach().numpy())
        # features: Tensor with shape [num_frames, feature_dim]
    result = np.array(results)
    labels = np.array(labels)
    return result, labels, recordings, timestamps

def prediction_Data2vec(test_dir):
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")
    results = []
    labels =[]
    recordings = []
    timestamps = []
    for batch, label, recording, start_time in test_dir:
        inputs = processor(batch, return_tensors="pt", padding=True, sampling_rate=16000)
        batch = inputs['input_values'].squeeze(0)
        result = model(batch)
        results.extend(torch.flatten(result['extract_features'], start_dim=1).detach().numpy())
        labels.extend(label.cpu().detach().numpy())
        recordings.extend(list(recording))
        timestamps.extend(start_time.cpu().detach().numpy())

    result = np.array(results)
    labels = np.array(labels)
    return result, labels, recordings, timestamps

def get_Bioacoustic_features(embedding_dir):
    labels = []
    features = []
    recording = []
    for label in os.listdir(embedding_dir):
        if label.endswith('.yml'):
            continue
        sub_dir = os.path.join(embedding_dir, label)
        for file in os.listdir(sub_dir):
            sub_array = np.load(os.path.join(sub_dir, file))
            features.extend(sub_array)
            labels.extend([label]*sub_array.shape[0])
            recording.extend([file] * sub_array.shape[0])
    return np.array(features), np.array(labels), np.array(recording), None







