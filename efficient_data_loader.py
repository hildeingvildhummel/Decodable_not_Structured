"""https://music-classification.github.io/tutorial/part3_supervised/tutorial.html"""
import os
from torch.utils import data
from torch.utils.data import Dataset
from random import choice
import math
import soundfile as sf
from scipy import signal
import numpy as np
import torch

class ReadAudioSections():

    def __init__(self, sample_len, samplerate):
        self.sample_len = sample_len
        self.sample_rate = samplerate
    def _read_audio_section(self, filename, start_time, stop_time, augmentation=False):
        track = sf.SoundFile(filename)

        can_seek = track.seekable()  # True
        if not can_seek:
            raise ValueError("Not compatible with seeking")

        sr = track.samplerate
        sample_len = stop_time - start_time
        start_frame = int(sr * start_time)

        frames_to_read = int(sr * (sample_len))

        track.seek(start_frame)
        audio_section = track.read(frames_to_read)
        if augmentation:
            new_time = choice([i for i in range(0, int(track.frames/sr) - sample_len) if i not in range(start_time,
                                                                                           start_time + sample_len)])
            track.seek(new_time)
            audio_section_aug = track.read(frames_to_read)
            return audio_section, sr, audio_section_aug
        return audio_section, sr

    def _reduce_sr(self, audio, sr, new_sr, sample_size):
        if sr == new_sr:
            return audio
        else:
            resampled = signal.resample(audio, int(new_sr * sample_size))
            return resampled

    def extract_as_clip(self, input_filename, start_time, stop_time, sample_size = None):
        if sample_size is None:
            sample_size = self.sample_len
        audio_extract, sr = self._read_audio_section(input_filename, start_time, stop_time)
        audio_extract = self._reduce_sr(audio_extract, sr, self.sample_rate, sample_size)
        audio_extract = torch.from_numpy(np.array(audio_extract, dtype='float32'))
        return audio_extract, sr

class DeepShipLoader():
    def __init__(self, sample_len):
        self.sample_len = sample_len

    def _extract_duration_audio(self, input_filename):
        track = sf.SoundFile(input_filename)
        num_frames = track.frames
        sr = track.samplerate
        return num_frames / (self.sample_len * sr)
        # data, sr = librosa.load(input_filename)
        # return len(data) / sr

    def extract_label_and_duration(self, filename):
        labels = {}
        for root, dirs, files in os.walk(filename):
            for input_filename in files:
                if input_filename.endswith('.wav'):
                    subdir = root.split('/')[-1]
                    time_ext = self._extract_duration_audio(os.path.join(root, input_filename))
                    # print(root, input_filename, time)
                    # if time < self.sample_len:
                    #     continue
                    duration = math.floor(time_ext)

                    labels[os.path.join(root, input_filename)] = [duration, subdir]
        return labels

    def extract_shuffled_label_and_duration(self, filename):
        import random
        labels = {}
        for root, dirs, files in os.walk(filename):
            all_labels = dirs
            print('all labels: ', all_labels)
            break
        for root, dirs, files in os.walk(filename):
            for input_filename in files:
                if input_filename.endswith('.wav'):
                    # print(input_filename)
                    # subdir = root.split('/')[-1]
                    # print(subdir)
                    subdir = random.choice(all_labels)
                    # print(subdir)
                    # print('++++++++++')
                    time_ext = self._extract_duration_audio(os.path.join(root, input_filename))
                    # print(root, input_filename, time)
                    # if time < self.sample_len:
                    #     continue
                    duration = math.floor(time_ext)

                    labels[os.path.join(root, input_filename)] = [duration, subdir]
        return labels

    def return_recording_index(self, label_dictionary):
        values = np.array(list(label_dictionary.values()))
        durations = values[:, 0].astype(int)
        index_list = []
        for idx, x in enumerate(durations):
            if idx == 0:
                index_list.append(0)
                continue
            previous_value = index_list[idx - 1]
            new_value = durations[idx - 1] + previous_value
            index_list.append(new_value)
        return index_list

    def get_total_duration(self, label_dictionary):
        values = np.array(list(label_dictionary.values()))
        total_duration = sum(values[:, 0].astype(int))
        return total_duration


class EfficientDataSet(Dataset):
    def __init__(self, parent_dir, sample_len, samplerate, return_recording, shuffled=False):
        super().__init__()
        self.parentdir = parent_dir
        self.sample_len = sample_len
        self.samplerate = samplerate
        self.Label_translation = self._get_Label_translation()
        self.index_functions = self._labeled_index_to_start_time
        self.labeled_fuctions = DeepShipLoader(self.sample_len)
        if not shuffled:
            self.label_dict = self.labeled_fuctions.extract_label_and_duration(self.parentdir)
        else:
            self.label_dict = self.labeled_fuctions.extract_shuffled_label_and_duration(self.parentdir)
        self.return_recording = return_recording

    def _get_Label_translation(self):
        label_dict = {}
        list_of_dirs = [f.name for f in os.scandir(self.parentdir) if f.is_dir()]
        counter = 0
        for class_name in list_of_dirs:
            label_dict[class_name] = counter
            counter += 1
        # print(label_dict)
        return label_dict

    def _extract_wav_files(self):
        list_of_dirs = self._get_subdirs(self.parentdir)
        wav_files = []
        for directory in list_of_dirs:
            wav_files.extend([os.path.join(directory, fi) for fi in os.listdir(directory) if fi.endswith(".wav")])
            wav_files.extend([os.path.join(directory, fi) for fi in os.listdir(directory) if fi.endswith(".flac")])
        return wav_files

    def _get_subdirs(self, directory):
        # print(directory)
        fu = [x[0] for x in os.walk(directory)]
        return fu
    def _labeled_index_to_start_time(self, index):
        # recordings = self._extract_wav_files()

        # label_dict = self.labeled_fuctions.extract_label_and_duration(self.parentdir)
        label_dict = self.label_dict
        recordings = list(label_dict.keys())

        recording_index = self.labeled_fuctions.return_recording_index(label_dict)
        index_of_interest = [i for i, e in enumerate(recording_index) if e <= index][-1]
        recording_of_interest = recordings[index_of_interest]
        # if index_of_interest !=0:
        previous_num_samples = 0
        for i in range(index_of_interest):
            previous_recording = recordings[i]
            previous_num_samples += label_dict[previous_recording][0]

        # start_time = (index - (index_of_interest * num_samples)) * self.sample_len
        start_time = (index - previous_num_samples) * self.sample_len

        return recording_of_interest, int(start_time)

    def _labeled_len_function(self):
        # label_dict = self.labeled_fuctions.extract_label_and_duration(self.parentdir)
        label_dict = self.label_dict
        durations = self.labeled_fuctions.get_total_duration(label_dict)
        return durations
    def __getitem__(self, item):
        recording, start_time = self.index_functions(item)
        # print(recording)
        read_audio = ReadAudioSections(self.sample_len, self.samplerate)

        audio, sr = read_audio.extract_as_clip(recording, start_time, start_time + self.sample_len)

        label_dict = self.label_dict
        # print(label_dict[recording][1])
        label = self.Label_translation[label_dict[recording][1]]
        # print(label)
        # print('++++++++')
        if self.return_recording:
            return audio, label, recording, start_time
        else:
            return audio, label
    def __len__(self):
        return self._labeled_len_function()
def get_dataloader(recording_path,
                   sample_len_sec=10, sample_rate=16000, batch_size=64, return_recording=False, shuffled=False):
    data_loader = data.DataLoader(dataset=EfficientDataSet(parent_dir=recording_path,
                                                              sample_len=sample_len_sec,
                                                              samplerate=sample_rate,
                                                           return_recording=return_recording, shuffled=shuffled),
                                      batch_size=batch_size,
                                      drop_last=False,
                                      num_workers=1)
    # print('elapsed time: ', time.time() - start_time)
    #
    # for item in data_loader:
    #     print(type(item))
    return data_loader