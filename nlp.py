#import library
import speech_recognition as sr
import os

"""
if BUTTON PRESSED == TRUE: 
    text = speech_to_text()
    if command_exist(text) == true
    
"""

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio_text = r.listen(source)
        try:
            return r.recognize_google(audio_text)
        except:
            return None

def command_exist(text):
    path = 'Test/'+text
    isFile = os.path.isdir(path)
    return isFile





