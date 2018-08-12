import numpy as np
from lxml import etree
import os.path
import librosa

import Input.Input
from Sample import Sample
#
# def data_load(path, dataset):
#     root = path
#     if dataset == 'train_unsup':
#         mix_list = glob.glob(root+dataset+'/*.wav')
#         voice_list = list()
#     else:
#         cd clob(root+dataset+'/Mixed/*.wav')
#         voice_list = glob.glob(root+dataset+'/Drums/*.wav')
#     mix = list()
#     voice = list()
#     for item in mix_list:
#         mix.append(Sample.from_path(item))
#     for item in voice_list:
#         voice.append(Sample.from_path(item))
#     return mix, voice

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





# TODO: Investigate ccmixter and difficulty of generating xml with drums instead of vocals
#
# def getCCMixter(xml_path):
#     tree = etree.parse(xml_path)
#     root = tree.getroot()
#     db_path = root.find("./databaseFolderPath").text
#     tracks = root.findall(".//track")
#
#     mixes, accs, vocals = list(), list(), list()
#
#     for track in tracks:
#         # Get mix and vocal instruments
#         voice = create_sample(db_path, track.xpath(".//instrument[instrumentName='Voice']")[0])
#         mix = create_sample(db_path, track.xpath(".//instrument[instrumentName='Mix']")[0])
#         acc = create_sample(db_path, track.xpath(".//instrument[instrumentName='Instrumental']")[0])
#
#         mixes.append(mix)
#         accs.append(acc)
#         vocals.append(voice)
#
#     return [mixes, accs, vocals]


# TODO: find or code MedleyDB code to replace this function getMedleyDB()
# def getMedleyDB(xml_path):
#     tree = etree.parse(xml_path)
#     root = tree.getroot()
#     db_path = root.find("./databaseFolderPath").text
#
#     mixes, accs, vocals = list(), list(), list()
#
#     tracks = root.xpath(".//track")
#     for track in tracks:
#         instrument_paths = list()
#         # Mix together vocals, if they exist
#         vocal_tracks = track.xpath(".//instrument[instrumentName='Voice']/relativeFilepath") + \
#                        track.xpath(".//instrument[instrumentName='Voice']/relativeFilepath") + \
#                        track.xpath(".//instrument[instrumentName='Voice']/relativeFilepath")
#         if len(vocal_tracks) > 0: # If there are vocals, get their file paths and mix them together
#             vocal_track = Input.Input.add_audio([db_path + os.path.sep + f.text for f in vocal_tracks], "vocalmix")
#             instrument_paths.append(vocal_track)
#             vocals.append(Sample.from_path(vocal_track))
#         else: # Otherwise append duration of track so silent input can be generated later on-the-fly
#             duration = float(track.xpath("./instrumentList/instrument/length")[0].text)
#             vocals.append(duration)
#
#         # Mix together accompaniment, if it exists
#         acc_tracks = track.xpath(".//instrument[not(instrumentName='Voice') and not(instrumentName='Mix') and not(instrumentName='Instrumental')]/relativeFilepath")
#         if len(acc_tracks) > 0:  # If there are vocals, get their file paths and mix them together
#             acc_track = Input.Input.add_audio([db_path + os.path.sep + f.text for f in acc_tracks], "accmix")
#             instrument_paths.append(acc_track)
#             accs.append(Sample.from_path(acc_track))
#         else:  # Otherwise append duration of track so silent input can be generated later on-the-fly
#             duration = float(track.xpath("./instrumentList/instrument/length")[0].text)
#             accs.append(duration)
#
#         # Mix together vocals and accompaniment
#         mix_track = Input.Input.add_audio(instrument_paths, "totalmix")
#         mixes.append(Sample.from_path(mix_track))
#
#     return [mixes, accs, vocals]

