import imp
import wave
import numpy as np
import os
import torch
import os.path
import torchaudio
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import librosa
from datasets import Audio

AUDIO_EXTENSIONS = ('.wav', '.mp3')

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}
    return classes, class_to_idx, idx_to_class

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    idx_to_class: Optional[Dict[str, int]] = None,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx or idx_to_class is None:
        _, class_to_idx, idx_to_class  = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if is_valid_file(fname):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances

class Datasets_Folder(object):
    def __init__(self,
            root: str,
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None):
        self.root = root
        self.extensions = extensions
        classes, class_to_idx, idx_to_class= self.find_classes(self.root)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        samples = self.make_dataset(self.root, class_to_idx, idx_to_class, extensions, is_valid_file)
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        idx_to_class:Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        return make_dataset(directory, class_to_idx, idx_to_class,extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self,index,use_torchaudio:bool=True,use_libsora:bool=False):
        file_audio_path, label = self.samples[index]

        # mfcc, fbank, original wav, 
        audio = Audio(file_audio_path)
        n_fft = 256
        win_length = None
        hop_length = 512
        n_mels = 64
        n_mfcc = 256
        n_fft = 1024
        waveform, sample_rate = torchaudio.load(file_audio_path)
        if use_torchaudio: 
            mfbank =  torchaudio.functional.melscale_fbanks(
                int(n_fft // 2 + 1),
                n_mels=n_mels,
                f_min=0.0,
                f_max=sample_rate / 2.0,
                sample_rate=sample_rate,
                norm="slaney",
            )
            melspec =torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm="slaney",
                onesided=True,
                n_mels=n_mels,
                mel_scale="htk",
            )(waveform)
            spectrogram = torchaudio.transforms.Spectrogram(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
            )(waveform)
            mfcc = torchaudio.transforms.MFCC(
                    sample_rate=sample_rate,
                    n_mfcc=n_mfcc,
                    melkwargs={
                        "n_fft": n_fft,
                        "n_mels": n_mels,
                        "hop_length": hop_length,
                        "mel_scale": "htk",
                    },
                )(waveform)
        elif use_libsora:
            mfbank = librosa.filters.mel(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                fmin=0.0,
                fmax=sample_rate / 2.0,
                norm="slaney",
                htk=True,
            ).T
            melspec = librosa.feature.melspectrogram(
                y=waveform.numpy()[0],
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                center=True,
                pad_mode="reflect",
                power=2.0,
                n_mels=n_mels,
                norm="slaney",
                htk=True,
            )
            mfcc = librosa.feature.mfcc(
                S=librosa.core.spectrum.power_to_db(melspec),
                n_mfcc=n_mfcc,
                dct_type=2,
                norm="ortho",
            )
        else:
            print(" At least one of [use_kaldi, use_torchaudio, use_libsora] is true, and use_kaldi > use_torchaudio > use_libsora")
        return audio, label, waveform, sample_rate,  mfbank, melspec, mfcc, 

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    data_dir = '/workspace/data/audio/CASIA'
    dataset = Datasets_Folder(root = data_dir,extensions='.wav')
    trainloder = torch.utils.data.DataLoader(
        dataset, 
        batch_size = 3,
        shuffle = True, 
        num_workers = 8,
        drop_last=False)
    print(len(dataset))
    for data in trainloder:
        print(data[0].shape) 
    
    # import random
    # # 随机查看五条音频
    # from IPython.display import Audio, display
    # for _ in range(5):
    #     rand_idx = random.randint(0, len(dataset)-1)
    #     example = ds["train"][rand_idx]
    #     audio = example["audio"]

    #     display(Audio(audio["array"], rate=audio["sampling_rate"]))
    #     print()
