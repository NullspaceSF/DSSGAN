import numpy as np
from lxml import etree
import os.path
import librosa
import Metadata
import subprocess
from soundfile import SoundFile

class Sample(object):
    '''
    Represents a particular audio track - maintains metadata about the audio file for faster audio handling during training
    '''
    def __init__(self, path, sample_rate, channels, duration):
        self.path = path
        self.sample_rate = sample_rate
        self.channels = channels
        self.duration = duration


    @classmethod
    def from_path(cls, path):
        '''
        Create new sample object from audio file path by retrieving metadata.
        :param path: 
        :return: 
        '''

        sr, channels, duration = Metadata.get_audio_metadata(path)
        return cls(path, sr, channels, duration)



def readWave(audio_path, start_frame, end_frame, mono=True, sample_rate=None, clip=True):
    snd_file = SoundFile(audio_path, mode='r')
    inf = snd_file._info
    audio_sr = inf.samplerate

    snd_file.seek(start_frame)
    audio = snd_file.read(end_frame - start_frame, dtype='float32')
    snd_file.close()
    audio = audio.T # Tuple to numpy, transpose axis to (channels, frames)

    # Convert to mono if desired
    if mono and len(audio.shape) > 1 and audio.shape[0] > 1:
        audio = np.mean(audio, axis=0)

    # Resample if needed
    if sample_rate is not None and sample_rate != audio_sr:
        audio = librosa.resample(audio, audio_sr, sample_rate, res_type="kaiser_fast")
        audio_sr = sample_rate

    # Clip to [-1,1] if desired
    if clip:
        audio = np.minimum(np.maximum(audio, -1.0), 1.0)

    return audio, audio_sr


def random_amplify(magnitude):
    '''
    Randomly amplifies or attenuates the input magnitudes
    :param magnitude: SINGLE Magnitude spectrogram excerpt, or list of spectrogram excerpts that each have their own amplification factor
    :return: Amplified magnitude spectrogram
    '''
    if isinstance(magnitude, np.ndarray):
        return np.random.uniform(0.2, 1.2) * magnitude
    else:
        assert(isinstance(magnitude, list))
        factor = np.random.uniform(0.2, 1.2)
        for i in range(len(magnitude)):
            magnitude[i] = factor * magnitude[i]
        return magnitude


def readAudio(audio_path, offset=0.0, duration=None, mono=True, sample_rate=None, clip=True, padding_duration=0.0, metadata=None):
    '''
    Reads an audio file wholly or partly, and optionally converts it to mono and changes sampling rate.
    By default, it loads the whole audio file. If the offset is set to None, the duration HAS to be not None,
    and the offset is then randomly determined so that a random section of the audio is selected with the desired duration.
    Optionally, the file can be zero-padded by a certain amount of seconds at the start and end before selecting this random section.
    :param audio_path: Path to audio file
    :param offset: Position in audio file (s) where to start reading. If None, duration has to be not None, and position will be randomly determined.
    :param duration: How many seconds of audio to read
    :param mono: Convert to mono after reading
    :param sample_rate: Convert to given sampling rate if given
    :param padding_duration: Amount of padding (s) on each side that needs to be filled up with silence if it isn't available
    :param metadata: metadata about audio file, accelerates reading audio since duration does not need to be determined from file 
    :return: Audio signal, Audio sample rate
    '''

    if os.path.splitext(audio_path)[1][1:].lower() == "mp3":  # If its an MP3, call ffmpeg with offset and duration parameters
        # Get mp3 metadata information and duration
        if metadata is None:
            audio_sr, audio_channels, audio_duration = Metadata.get_mp3_metadata(audio_path)
        else:
            audio_sr = metadata[0]
            audio_channels = metadata[1]
            audio_duration = metadata[2]
        print(audio_duration)

        pad_front_duration = 0.0
        pad_back_duration = 0.0

        if offset is None:  # In this case, select random section of audio file
            assert (duration is not None)
            max_start_pos = audio_duration+2*padding_duration-duration
            if (max_start_pos <= 0.0):  # If audio file is longer than duration of desired section, take all of it, will be padded later
                print("WARNING: Audio file " + audio_path + " has length " + str(audio_duration) + " but is expected to be at least " + str(duration))
                return librosa.load(audio_path, sample_rate, mono, res_type='kaiser_fast')  # Return whole audio file
            start_pos = np.random.uniform(0.0,max_start_pos) # Otherwise randomly determine audio section, taking padding on both sides into account
            offset = max(start_pos - padding_duration, 0.0) # Read from this position in audio file
            pad_front_duration = max(padding_duration - start_pos, 0.0)
        assert (offset is not None)

        if duration is not None: # Adjust duration if it overlaps with end of track
            pad_back_duration = max(offset + duration - audio_duration, 0.0)
            duration = duration - pad_front_duration - pad_back_duration # Subtract padding from the amount we have to read from file
        else: # None duration: Read from offset to end of file
            duration = audio_duration - offset

        pad_front_frames = int(pad_front_duration * float(audio_sr))
        pad_back_frames = int(pad_back_duration * float(audio_sr))


        args = ['ffmpeg', '-noaccurate_seek',
                '-ss', str(offset),
                '-t', str(duration),
                '-i', audio_path,
                '-f', 's16le', '-']

        audio = []
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=open(os.devnull, 'wb'))
        num_reads = 0
        while True:
            output = process.stdout.read(4096)
            if output == '' and process.poll() is not None:
                break
            if output:
                audio.append(librosa.util.buf_to_float(output, dtype=np.float32))
                num_reads += 1

        audio = np.concatenate(audio)
        if audio_channels > 1:
            audio = audio.reshape((-1, audio_channels)).T

    else: #Not an MP3: Handle with PySoundFile
        # open audio file
        snd_file = SoundFile(audio_path, mode='r')
        inf = snd_file._info
        audio_sr = inf.samplerate

        if duration is not None:
            num_frames = int(duration * float(audio_sr))
        pad_frames = int(padding_duration * float(audio_sr))
        pad_front_frames = 0
        pad_back_frames = 0

        if offset is None:  # In this case, select random section of audio file
            assert (duration is not None)
            max_start_pos = inf.frames + 2 * pad_frames - num_frames
            if (max_start_pos <= 0):  # If audio file is longer than duration of desired section, take all of it, will be padded later
                print("WARNING: Audio file " + audio_path + " has frames  " + str(inf.frames) + " but is expected to be at least " + str(num_frames))
                return librosa.load(audio_path, sample_rate, mono, res_type='kaiser_fast')  # Return whole audio file
            start_pos = np.random.randint(0, max_start_pos)  # Otherwise randomly determine audio section, taking padding on both sides into account
            start_frame = max(start_pos - pad_frames, 0)  # Read from this position in audio file
            pad_front_frames = max(pad_frames - start_pos, 0)
        else:
            start_frame = int(offset * float(audio_sr))

        if duration is not None:  # Adjust duration if it overlaps with end of track
            pad_back_frames = max(start_frame + num_frames - inf.frames, 0)
            num_frames = num_frames - pad_front_frames - pad_back_frames
        else: # Duration is None => Read from start frame to end of track
            num_frames = inf.frames - start_frame

        snd_file.seek(start_frame)
        audio = snd_file.read(num_frames, dtype='float32')
        snd_file.close()
        audio = audio.T  # Tuple to numpy, transpose axis to (channels, frames)

        centre_start_frame = start_frame - pad_front_frames + pad_frames
        centre_end_frame = start_frame + num_frames + pad_back_frames - pad_frames

    # AT THIS POINT WE HAVE A [N_CHANNELS, N_SAMPLES] NUMPY ARRAY FOR THE AUDIO
    # Pad as indicated at beginning and end
    if len(audio.shape) > 1:
        audio = np.pad(audio, [(0,0),(pad_front_frames, pad_back_frames)],mode="constant",constant_values=0.0)
    else:
        audio = np.pad(audio, [(pad_front_frames, pad_back_frames)], mode="constant", constant_values=0.0)

    # Convert to mono if desired
    if mono and len(audio.shape) > 1 and audio.shape[0] > 1:
        audio = np.mean(audio, axis=0)

    # Resample if needed
    if sample_rate is not None and sample_rate != audio_sr:
        audio = librosa.resample(audio, audio_sr, sample_rate, res_type="kaiser_fast")
        audio_sr = sample_rate

    # Clip to [-1,1] if desired
    if clip:
        audio = np.minimum(np.maximum(audio, -1.0), 1.0)

    if float(audio.shape[0])/float(sample_rate) < 1.0:
        print("----------------------ERROR------------------")

    if os.path.splitext(audio_path)[1][1:].lower() == "mp3":
        return audio, audio_sr
    else:
        return audio, audio_sr, centre_start_frame, centre_end_frame

# Return a 2d numpy array of the spectrogram
def audioFileToSpectrogram(audioIn, fftWindowSize=1024, hopSize=512, offset=0.0, duration=None, expected_sr=None, buffer=False, padding_duration=0.0, metadata=None):
    '''
    Audio to FFT magnitude and phase conversion. Input can be a filepath to an audio file or a numpy array directly.
    By default, the whole audio is used for conversion. By setting duration to the desired number of seconds to be read from the audio file,
    reading can be sped up.
    For accelerating reading, the buffer option can be activated so that a numpy filedump of the magnitudes
    and phases is created after processing and loaded the next time it is requested.
    :param audioIn: 
    :param fftWindowSize: 
    :param hopSize: 
    :param offset: 
    :param duration: 
    :param expected_sr: 
    :param buffer: 
    :return: 
    '''

    writeNumpy = False
    if isinstance(audioIn, str): # Read from file
        if buffer and os.path.exists(audioIn + ".npy"): # Do we need to load a previous numpy buffer file?
            assert(offset == 0.0 and duration is None) # We can only load the whole buffer file
            with open(audioIn + ".npy", 'r') as file: # Try loading
                try:
                    [magnitude, phase] = np.load(file)
                    return magnitude, phase
                except Exception as e: # In case loading did not work, remember and overwrite file later
                    print("Could not load " + audioIn + ".npy. Loading audio again and recreating npy file!")
                    writeNumpy = True
        audio, sample_rate, _ , _= readAudio(audioIn, duration=duration, offset=offset, sample_rate=expected_sr, padding_duration=padding_duration, metadata=metadata) # If no buffering, read audio file
    else: # Input is already a numpy array
        assert(expected_sr is None and duration is None and offset == 0.0) # Make sure no other options are active
        audio = audioIn

    # Compute magnitude and phase
    spectrogram = librosa.stft(audio, fftWindowSize, hopSize)
    magnitude, phase = librosa.core.magphase(spectrogram)
    phase = np.angle(phase) # from e^(1j * phi) to phi
    assert(np.max(magnitude) < fftWindowSize and np.min(magnitude) >= 0.0)

    # Buffer results if desired
    if (buffer and ((not os.path.exists(audioIn + ".npy")) or  writeNumpy)):
        np.save(audioIn + ".npy", [magnitude, phase])

    return magnitude, phase

def add_audio(audio_list, path_postfix):
    '''
    Reads in a list of audio files, sums their signals, and saves them in new audio file which is named after the first audio file plus a given postfix string
    :param audio_list: List of audio file paths
    :param path_postfix: Name to append to the first given audio file path in audio_list which is then used as save destination
    :return: Audio file path where the sum signal was saved
    '''
    save_path = audio_list[0] + "_" + path_postfix + ".wav"
    if not os.path.exists(save_path):
        for idx, instrument in enumerate(audio_list):
            instrument_audio, sr = librosa.load(instrument, sr=None)
            if idx == 0:
                audio = instrument_audio
            else:
                audio += instrument_audio
        if np.min(audio) < -1.0 or np.max(audio) > 1.0:
            print("WARNING: Mixing tracks together caused the result to have sample values outside of [-1,1]. Clipping those values")
            audio = np.minimum(np.maximum(audio, -1.0), 1.0)

        librosa.output.write_wav(save_path, audio, sr)
    return save_path

def subtract_audio(mix_list, instrument_list):
    '''
    Generates new audio by subtracting the audio signal of an instrument recording from a mixture
    :param mix_list: 
    :param instrument_list: 
    :return: 
    '''

    assert(len(mix_list) == len(instrument_list))
    new_audio_list = list()

    for i in range(0, len(mix_list)):
        new_audio_path = os.path.dirname(mix_list[i]) + os.path.sep + "remainingmix" + os.path.splitext(mix_list[i])[1]
        new_audio_list.append(new_audio_path)

        if os.path.exists(new_audio_path):
            continue
        mix_audio, mix_sr = librosa.load(mix_list[i], mono=False, sr=None)
        inst_audio, inst_sr = librosa.load(instrument_list[i], mono=False, sr=None)
        assert (mix_sr == inst_sr)
        new_audio = mix_audio - inst_audio
        if not (np.min(new_audio) >= -1.0 and np.max(new_audio) <= 1.0):
            print("Warning: Audio for mix " + str(new_audio_path) + " exceeds [-1,1] float range!")

        librosa.output.write_wav(new_audio_path, new_audio, mix_sr) #TODO switch to compressed writing
        print("Wrote accompaniment for song " + mix_list[i])
    return new_audio_list

def create_sample(db_path, instrument_node):
   path = db_path + os.path.sep + instrument_node.xpath("./relativeFilepath")[0].text
   sample_rate = int(instrument_node.xpath("./sampleRate")[0].text)
   channels = int(instrument_node.xpath("./numChannels")[0].text)
   duration = float(instrument_node.xpath("./length")[0].text)
   return Sample(path, sample_rate, channels, duration)

def getDSDFilelist(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    db_path = root.find("./databaseFolderPath").text
    tracks = root.findall(".//track")

    train_vocals, test_vocals, train_mixes, test_mixes, train_accs, test_accs = list(), list(), list(), list(), list(), list()

    for track in tracks:
        # Get mix and vocal instruments
        vocals = create_sample(db_path, track.xpath(".//instrument[instrumentName='Voice']")[0])
        mix = create_sample(db_path, track.xpath(".//instrument[instrumentName='Mix']")[0])
        [acc_path] = subtract_audio([mix.path], [vocals.path])
        acc = Sample(acc_path, vocals.sample_rate, vocals.channels, vocals.duration) # Accompaniment has same signal properties as vocals and mix

        if track.xpath("./databaseSplit")[0].text == "Training":
            train_vocals.append(vocals)
            train_mixes.append(mix)
            train_accs.append(acc)
        else:
            test_vocals.append(vocals)
            test_mixes.append(mix)
            test_accs.append(acc)

    return [train_mixes, train_accs, train_vocals], [test_mixes, test_accs, test_vocals]
