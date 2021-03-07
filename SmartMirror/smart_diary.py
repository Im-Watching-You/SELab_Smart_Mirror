from __future__ import print_function

import os

from ibm_watson import ToneAnalyzerV3, LanguageTranslatorV3
import pyaudio
import wave
import time
from threading import Thread
from flags import ButtonFlag
from google.cloud import speech_v1p1beta1 as speech
from db_connector import DiarySession
import datetime
import json


class Smart_Diary:
    def __init__(self):
        self.rec = Recorder(channels=2)
        self.stt = DiaryAnalysis()
        self.session = DiarySession()
        self.num = 0
        self.user_id = -1
        self.language = 'ko-KR_BroadbandModel'

        self.session_list = []

    def set_user(self, id):
        self.user_id = id

    def record_start(self):
        def rs():
            session_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            file_name_2 = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            file_name = '.\\diary\\{}.wav'.format(file_name_2)
            with self.rec.open(file_name, 'wb') as recfile:
                recfile.start_recording()
                while ButtonFlag.record_flag:
                    time.sleep(0.5)
                recfile.stop_recording()
            result = self.stt.speech_to_text(file_name)
            if result:
                result_txt = result
                text_path = '.\\diary\\content\\{}.txt'.format(file_name_2)
                f = open(text_path, 'w')
                f.write(result_txt)
                f.close()
                print(result_txt)
                language, ko = self.stt.identify_language(result_txt)
                if ko:
                    result_txt = self.stt.trans_to_eng(result_txt, language)
                emotion = self.stt.tone_analyzer(result_txt)
                print(emotion)
                data = dict()
                data['session_time'] = session_time
                data['file_name'] = file_name
                data['content'] = text_path
                data['emotion'] = emotion
                print(data)
                self.insert_info_db(self.user_id, data)
        Thread(target=rs).start()

    def insert_info_db(self, id, data):
        self.session.insert_diary_session(id, data)

    def update_info_db(self, id):
        self.session_list = self.session.load_diary_session(id)

    def get_diary_data(self):
        return self.session_list


class Recorder(object):
    '''A recorder class for recording audio to a WAV file.
    Records in mono by default.
    '''

    def __init__(self, channels=1, rate=44100, frames_per_buffer=1024):
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer

    def open(self, fname, mode='wb'):
        return RecordingFile(fname, mode, self.channels, self.rate,
                            self.frames_per_buffer)


class RecordingFile(object):
    def __init__(self, fname, mode, channels,
                rate, frames_per_buffer):
        self.fname = fname
        self.mode = mode
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self._pa = pyaudio.PyAudio()
        self.wavefile = self._prepare_file(self.fname, self.mode)
        self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        self.close()

    def record(self, duration):
        # Use a stream with no callback function in blocking mode
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.frames_per_buffer)
        for _ in range(int(self.rate / self.frames_per_buffer * duration)):
            audio = self._stream.read(self.frames_per_buffer)
            self.wavefile.writeframes(audio)
        return None

    def start_recording(self):
        # Use a stream with a callback in non-blocking mode
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.frames_per_buffer,
                                        stream_callback=self.get_callback())
        self._stream.start_stream()
        return self

    def stop_recording(self):
        self._stream.stop_stream()
        return self

    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            self.wavefile.writeframes(in_data)
            return in_data, pyaudio.paContinue
        return callback


    def close(self):
        self._stream.close()
        self._pa.terminate()
        self.wavefile.close()

    def _prepare_file(self, fname, mode='wb'):
        wavefile = wave.open(fname, mode)
        wavefile.setnchannels(self.channels)
        wavefile.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        wavefile.setframerate(self.rate)
        return wavefile


class DiaryAnalysis:
    def __init__(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./smart-mirror-system-9bc6ecf8d44d.json"
        self.stt_client = speech.SpeechClient()
        self.Tone_service = ToneAnalyzerV3(
            url='https://gateway-tok.watsonplatform.net/tone-analyzer/api',
            version='2017-09-21',
            iam_apikey='Qz0WbzSPLe1y1jVo2OxFlvPHCu_fA9LCPamFtNaliWrV'
        )
        self.language_translator = LanguageTranslatorV3(
            version='2018-05-01',
            iam_apikey="ri7SLZMdD2Lr8xygdKSGQ0cceONkzuWwv8Aw67vSAXC7",
            url="https://gateway-tok.watsonplatform.net/language-translator/api"
        )

    def speech_to_text(self, file_path):
        with open(file_path, 'rb') as audio_file:
            content = audio_file.read()
        audio = speech.types.RecognitionAudio(content=content)
        config = speech.types.RecognitionConfig(
            encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            audio_channel_count=2,
            language_code='en-US',
            alternative_language_codes=['ko-KR']
        )
        response = self.stt_client.recognize(config, audio)
        result_txt = ""
        for i, result in enumerate(response.results):
            alternative = result.alternatives[0]
            result_txt = result_txt+ alternative.transcript
        print(result_txt)
        return result_txt


    def identify_language(self, text):
        ko = False
        language = self.language_translator.identify(
            text).get_result()
        la = language['languages'][0]['language']
        print(json.dumps(language['languages'][0], indent=2))
        if not la == 'en':
            ko = True
        return la, ko

    def trans_to_eng(self, text, language):
        translation = self.language_translator.translate(
            text=text,
            target='en',
            source=language
        ).get_result()
        result = translation['translations'][0]['translation']
        print(result)
        return result

    def tone_analyzer(self, text):
        result = self.Tone_service.tone(tone_input=text, content_type="text/plain").get_result()
        eva = None
        print(result)
        if result['document_tone']['tones']:
            eva = result['document_tone']['tones']
        else:
            for a in result['sentences_tone']:
                if a['tones']:
                    eva = a['tones']
                else:
                    continue

        print(eva)
        if eva:
            for i in range(0, len(eva)):
                eva[i]['score'] = str(int(eva[i]['score']*100))
            print(eva[0]['score'])
        else:
            eva = [{'tone_name': " ", 'score':" "}]
        return eva


if __name__ == '__main__':
    pass
    stt = DiaryAnalysis()
    txt = stt.speech_to_text('./diary/2019_05_29_10_05_22.wav')
    print(txt)
