# -*- coding: utf-8 -*-
"""
transcription/fast_whisper3.py

Speaker-diarized transcription script with Human/Machine (AMD) labeling.
Loops through all audio files in a folder and saves transcripts in real time.

REQUIREMENTS  (install once in your venv)

    pip install openai-whisper pyannote.audio==3.1.1 numpy==1.26.4 \
                pandas==2.2.2 scikit-learn torch torchaudio

FFmpeg must be installed and on PATH:
    https://ffmpeg.org/download.html

OUTPUT FORMAT  (one combined transcript .txt per run)

    SPEAKER 1 0:00:00
    Mr. Jeff. Hi, good morning. This is Anna..._machine

    SPEAKER 2 0:00:17
    I have dentures._human
"""

import os
import sys
import subprocess
import datetime
import wave
import contextlib
import traceback

import torch
import numpy as np
import whisper

from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering

# ========== CONFIGURATION (hardcoded for this run) ==========
# These values replace the original config.py import
TRANSCRIPTION = {
    'num_speakers': 2,                        # number of speakers to diarize
    'language': 'English',                     # whisper language
    'model_size': 'base',                      # whisper model size (tiny/base/small/medium/large)
    'supported_extensions': ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg'],
    'output_file_name': 'combined_transcript.txt'   # name of the single output file
}

NUM_SPEAKERS = TRANSCRIPTION['num_speakers']
LANGUAGE     = TRANSCRIPTION['language']
MODEL_SIZE   = TRANSCRIPTION['model_size']
SUPPORTED_EXTS = TRANSCRIPTION['supported_extensions']

# ========== INPUT / OUTPUT PATHS (modified as requested) ==========
AUDIO_FOLDER  = "data/recordings_all3"      # folder containing audio files to transcribe
OUTPUT_FOLDER = "data"                       # folder where combined transcript will be saved

# ========== AMD keyword lists (unchanged) ==========

HUMAN_KEYWORDS = [
    # Greetings
    "hello", "hi", "hey", "yes", "yeah", "yep", "yup", "speaking",
    "this is", "who is this", "who's calling", "who are you",
    # Questions/Confusion
    "what", "huh", "excuse me", "sorry", "pardon",
    # Time-based greetings
    "good morning", "good afternoon", "good evening",
    # Conversational
    "how are you", "how can i help", "may i help you",
    "can you hear me", "are you there", "i can hear you",
    # Question words (short)
    "who", "when", "where", "why", "how",
    # Affirmations
    "okay", "ok", "sure", "alright", "fine", "right",
    # Hold requests
    "hold on", "wait", "one moment", "just a second", "one sec",
    # Names/Identity
    "speaking", "residence", "can i help","speaking", "this is", "how can i help",
    "yeah", "yes", "okay", "just a sec","speaking","this is","how can i help","yeah","yes","okay","just a sec","yes", "speaking",
    "this is", "how can i help", "who is this",
     # AFFIRMATIVE RESPONSES
    "yes", "yeah", "yep", "yup", "sure", "right", "all right", "alright",
    "correct", "ok", "okay", "absolutely", "uh huh", "uh", "oh", "um", "well", "huh", "hmm", "mm hmm",
    # PROFANITY
    "fuck", "fuck you", "bitch", "bastard", "motherfucker", "fucker",
    "asshole", "go fuck yourself", "fuck off", "bullshit",

    # INSULTS
    "idiot", "stupid", "dumbass", "dumb", "nonsense",

    # CALL REJECTION PHRASES
    "stop calling me", "stop calling", "don't call me again", "don't call again",
    "don't call me", "don't call", "quit calling me", "quit calling",
    "leave me alone", "get lost", "go away", "shut up",
    "never call", "stop harassing me", "stop harassing", "harassing me",
    "cease calling", "halt your calls", "end these calls", "no more calls",
    "no more phone calls", "I don't want any more calls", "don't ever call me",
    "don't contact me again", "remove me from your list", "take me off your list",
    "unsubscribe me", "I'll report you", "I'll call the police", "I'll sue you",
    "this is harassment", "I'm reporting this number", "I'm blocking you",
    "I'm filing a complaint", "I'm not interested", "not interested",
    "no interest", "I don't want this", "I don't want your calls",
    "I refuse these calls", "I reject your calls", "screw off", "buzz off",
    "piss off", "drop dead", "please cease all contact", "terminate all communications",
    "discontinue calling this number", "refrain from future contact",
    "I wish to be placed on your do-not-call list", "who is this?",
    "what company is this?", "where did you get this number?", "how did you get this number?",
    "stop or I'll report you", "identify yourself", "I'm hanging up now",
    "goodbye forever", "this conversation is over", "I'm ending this call",
    "stop calling me, please", "please don't call again", "I'm asking you to stop",
    "I'm telling you to stop", "this is your last warning", "final warning",
    "do-not-call list", "violating do-not-call", "TCPA violation",
    "telemarketing violation", "illegal robocall",  "fuck", "fuck you", "bitch", "bastard", "motherfucker", "fucker",
    "asshole", "go fuck yourself", "fuck off", "bullshit",

    # INSULTS
    "idiot", "stupid", "dumbass", "dumb", "nonsense",

    # CALL REJECTION PHRASES
    "stop calling me", "stop calling", "don't call me again", "don't call again",
    "don't call me", "don't call", "quit calling me", "quit calling",
    "leave me alone", "get lost", "go away", "shut up",
    "never call", "stop harassing me", "stop harassing", "harassing me",
    "cease calling", "halt your calls", "end these calls", "no more calls",
    "no more phone calls", "I don't want any more calls", "don't ever call me",
    "don't contact me again", "remove me from your list", "take me off your list",
    "unsubscribe me", "I'll report you", "I'll call the police", "I'll sue you",
    "this is harassment", "I'm reporting this number", "I'm blocking you",
    "I'm filing a complaint", "I'm not interested", "not interested",
    "no interest", "I don't want this", "I don't want your calls",
    "I refuse these calls", "I reject your calls", "screw off", "buzz off",
    "piss off", "drop dead", "please cease all contact", "terminate all communications",
    "discontinue calling this number", "refrain from future contact",
    "I wish to be placed on your do-not-call list", "who is this?",
    "what company is this?", "where did you get this number?", "how did you get this number?",
    "stop or I'll report you", "identify yourself", "I'm hanging up now",
    "goodbye forever", "this conversation is over", "I'm ending this call",
    "stop calling me, please", "please don't call again", "I'm asking you to stop",
    "I'm telling you to stop", "this is your last warning", "final warning",
    "do-not-call list", "violating do-not-call", "TCPA violation",
    "telemarketing violation", "illegal robocall"
]

MACHINE_KEYWORDS = [
    # Voicemail indicators
    "leave a message", "after the beep", "after the tone","please drop a message.","are you enrolled in part?","are you currently enrolled in?",
    "not available", "cannot take your call", "can't take your call",
    "please leave", "you have reached", "you've reached",
    "the person you are calling", "the person you have called",
    "is not available", "isn't available", "unavailable",
    "at the tone", "after the signal",
    "voicemail", "voice mail", "answering machine",
    "mailbox", "record your message", "leave your message",
    # IVR/Menu indicators
    "press", "dial", "extension", "directory","either",
    "for sales", "for support", "for billing",
    "press 1", "press 2", "press 0", "press star", "press pound",
    "to reach", "to leave a message", "to speak to",
    # Sound indicators
    "beep", "tone", "signal",
    # Time-based
    "office hours", "business hours", "currently closed",
    # Thank you messages (usually machine)
    "thank you for calling", "thanks for calling","pound key", "press pound", "record", "please leave", "leave a message",
    "can't come", "can't get", "forwarded", "sorry i miss", "sorry i missed",
    "we are not", "not available", "we're unable", "we are unable",
    "can't take", "can't answer", "voice mail", "voicemail", "leave your name",
    "after the tone", "at the tone", "beep",  "pound key","press pound","record","please leave","leave a message",
    "can't come","can't get","forwarded","sorry i miss","sorry i missed",
    "we are not","not available","we're unable","we are unable",
    "can't take","can't answer","voice mail","voicemail","leave your name",
    "after the tone","at the tone","beep","leave a message","after the tone","after the beep","press 1","press pound",
    "voicemail","mailbox","not available","sorry i missed","can't take your call",
    "your call is important","please hold","press any key","automated system",
    "ivr","this call may be recorded","return your call","business hours",
    "office is closed","welcome to","thank you for calling","leave a message", "leave your message", "leave me a message",
    "leave a brief message", "leave your name", "leave your number",
    "leave your name and number", "leave your name and phone number",
    "please leave", "please record", "record your message",
    "start speaking", "begin recording", "message recorded", "message saved",
    "please record your message", "at the tone please record",
    "tone please record your", "tone please record",
    "record your message when", "when you've finished recording",
    "when you have finished recording", "when you are finished recording",
    "finished recording you may", "recording you may hang",
    "you may hang up or press", "you may hang up or", "you may hang up",
    "may hang up or press", "hang up or press pound", "hang up or press 1",
    "hang up or press", "simply hang up or press", "simply hang up or",
    "simply hang up", "recording simply hang", "please leave your name and",
    "please leave a message", "leave your name and", "leave a message and",
    "please leave your message after", "please leave a brief message",
    "please leave a message after", "please leave me a message",
    "leave your name and phone number", "leave your name and a",
    "leave your message after the", "leave your message after",
    "leave a message after the", "leave a message after",
    "leave me a message and", "your message when you", "your message when",
    "message when you've", "message when you have", "message when you",
    "finished recording", "when you have finished", "when you've finished",
    "you have finished", "have finished recording", "up or press", "or press"," return the call thank you",
    "please press 1 to connect your call",

    # TONE & BEEP INDICATORS
    "at the tone", "after the tone", "after the beep", "beep",
    "at the tone please", "after the tone please", "message after the tone",
    "message after the beep", "the beep we will", "the beep",
    "the tone please", "the tone", "at the sound of the tone", "at the sound of the beep",
    "please speak after the tone", "please speak after the beep",
    "record a message at the tone", "record a message at the beep",
    "when you are done recording", "when you are done",
    "end of message", "to end your recording", "to end recording",
    "to finish recording", "to complete your message",
    "your message will be recorded", "your message will be saved",
    "to save your message", "to review your message",
    "press star to review", "press pound to save",
    "press any key to continue", "press any key",
    "wait for the prompt", "wait for the tone",
    "please wait for the tone", "please wait for the beep",
    "recording will begin after", "recording begins after",
    "you may begin speaking at the tone", "you may begin speaking at the beep",
    "begin speaking at the tone", "begin speaking at the beep",
    "start talking at the tone", "start talking at the beep",
    "at the sound", "after the sound", "sound of the tone",
    "sound of the beep", "you will hear a tone", "you will hear a beep",
    "listen for the tone", "listen for the beep",
    "tone will sound", "beep will sound", "short tone",
    "short beep", "long tone", "long beep",

    # IVR NAVIGATION
    "press 0 for operator", "press 0 for assistance",
    "press 9 for directory", "press star for main menu",
    "press star to return", "press star to repeat",
    "to repeat this message", "to hear these options again",
    "for spanish press 2", "para español marque dos",
    "for other languages", "language options",
    "if this is an emergency", "in case of emergency",
    "to make a payment", "for billing inquiries",
    "for technical support", "for sales department",
    "to schedule an appointment", "to cancel an appointment",
    "to speak with a representative", "to speak with someone",
    "to reach a live person", "to talk to a person",

    # UNAVAILABILITY & STATUS
    "is not available", "not available", "unavailable", "not able to come",
    "unable to take your call", "unable to answer", "we are unable",
    "we are not", "can't take your call", "can't answer", "can't come",
    "can't get", "sorry i missed", "sorry i am not available",
    "i'm sorry i missed your call", "we missed your call", "forwarded",
    "nothing has been recorded", "is not available at the tone",
    "is not available at this time", "is not available at",
    "not available at the tone", "not available at the",
    "not available at this", "available at the tone",
    "is not available please", "can't take your call now at",
    "can't take your call now", "cannot take your call",
    "take your call now at the", "take your call now at", "take your call now",
    "currently unavailable", "temporarily unavailable",
    "out of the office", "away from desk", "away from phone",
    "on another call", "on the phone", "busy line",
    "line is busy", "please try again later", "please call back later",
    "office is closed", "office hours are", "business hours",
    "after hours", "holiday schedule", "vacation message",
    "out of town", "traveling", "on vacation", "on leave",
    "on sick leave", "medical leave", "maternity leave",
    "paternity leave", "will be back", "return on",

    # MAILBOX & SYSTEM MESSAGES
    "mailbox", "voice mailbox", "voicemail", "mailbox is full",
    "voicemail is full", "the mailbox is full", "cannot accept messages",
    "cannot accept any messages", "is full and cannot accept new messages",
    "voice messaging system", "has not been set up yet goodbye",
    "has not been set up yet", "not been set up yet", "not been set up",
    "mailbox is full and cannot accept", "mailbox is full and cannot",
    "is full and cannot accept new", "is full and cannot accept",
    "cannot accept new messages", "voice mailbox that has not",
    "voice message system", "try again later goodbye",
    "mailbox has not been initialized", "mailbox not configured",
    "voicemail not set up", "personal greeting not set",
    "system greeting", "default greeting", "generic greeting",
    "this number is not in service", "number disconnected",
    "number has been changed", "new number is",
    "this number is no longer in use", "out of service",
    "circuit busy", "all circuits are busy",
    "network busy", "system busy", "high call volume",
    "unexpected error", "system error", "technical difficulties",
    "please try your call again", "unable to complete your call",

    # CALLBACK PROMISES
    "return your call", "i will return your call", "we will return your call",
    "call you back", "i will call you back", "call me back",
    "get back to you", "i'll get back to you", "as soon as possible",
    "as soon as you can", "right now", "will return your call as soon",
    "return your call as soon", "i'll get back to you as soon",
    "get back to you as soon as", "get back to you as soon",
    "get back with you as soon", "get back with you",
    "call you back as soon", "as soon as possible thank you",
    "soon as possible thank", "soon as possible",
    "will get back to you", "shall return your call",
    "expect a callback", "expect a return call",
    "i'll try you back", "we'll try you back",
    "at my earliest convenience", "at our earliest convenience",
    "when i return", "when we return", "upon my return",
    "as soon as i can", "as soon as we can",
    "at the next opportunity", "next chance i get",
    "when possible", "when i'm available",

    # SCAM/ROBOCALL PATTERNS
    "verification required", "immediate attention required", "legal action",
    "support advisor", "unauthorized", "fraud", "medicare", "benefits",
    "qualify", "amazon", "shopify", "ebay", "to send a message",
    "to send an sms notification", "para continuar en español",
    "screened by smart call blocker", "smart call blocker",
    "say cancel or press", "your call",
    "urgent matter", "important information", "final notice",
    "account suspended", "account compromised", "security alert",
    "warranty expired", "warranty about to expire",
    "credit card offer", "loan approval", "debt consolidation",
    "student loan forgiveness", "social security",
    "irs", "internal revenue service", "tax debt",
    "free trial", "special offer", "limited time",
    "prize winner", "you have won", "congratulations you",
    "tech support", "microsoft support", "apple support",
    "windows support", "virus detected", "malware alert",
    "suspicious activity", "unusual login attempt",
    "press 1 to accept", "press 1 to claim",
    "press 1 to speak with", "press 1 now",
    "to be removed", "to unsubscribe",
    "do not press any keys", "do not hang up",

    # TELEMARKETING PHRASES
    "survey", "market research", "opinion poll",
    "political survey", "charity donation",
    "fundraising", "non-profit organization",
    "sales call", "product demonstration",
    "free estimate", "free consultation",
    "no obligation", "no cost",
    "exclusive offer", "pre-approved",

    # BUSINESS SPECIFIC
    "doctor's office", "medical practice",
    "dental office", "veterinary clinic",
    "law office", "attorney at law",
    "real estate office", "insurance agency",
    "financial advisor", "investment firm",
    "property management", "home services",
    "utility company", "cable company",
    "internet provider", "phone company",
    "bank", "credit union", "financial institution",

    # TIME AND DATE REFERENCES
    "office hours", "business hours", "open from",
    "closed on", "weekends", "holidays",
    "eastern time", "pacific time", "central time",
    "mountain time", "standard time", "daylight time",
    "today is", "current date", "current time",

    # SYSTEM PROMPTS
    "please enter", "please say", "please speak",
    "using your keypad", "using your voice",
    "followed by the pound key", "followed by pound",
    "then press pound", "then press hash",
    "to confirm press", "to verify press",
    "to continue press", "to proceed press",
    "to opt out", "to stop receiving calls",

    # CONNECTION MESSAGES
    "transferring", "connecting", "redirecting",
    "please hold while", "one moment please",
    "momentarily", "shortly", "briefly",
    "your call is important to us",
    "all agents are currently busy",
    "next available representative",

    # IVR GREETINGS
    "you've reached", "thank you for calling", "have a blessed day",
    "have a great day", "have a wonderful day", "have a good day",
    "thank you and have a", "you've reached the voicemail", "you have reached",
    "sorry i missed your call please", "sorry we missed your call",
    "i missed your call",
    "welcome to", "thank you for contacting",
    "thanks for calling", "appreciate your call",
    "good day", "take care", "talk to you soon",
    "looking forward to", "speak with you soon",
    "have a nice day", "have a productive day",
    "best regards", "warm regards", "sincerely",
    "respectfully", "yours truly",

    # IVR PHRASES
    "welcome to our automated system", "automated attendant","license agency",
    "interactive voice response", "ivr system",
    "for quality assurance", "for training purposes",
    "calls are monitored", "calls are recorded",
    "your call will be answered", "in the order it was received",
    "estimated wait time", "current wait time",
    "you are number", "in the queue",
    "please remain on the line", "please do not hang up",
    "your call is being transferred", "transferring your call",
    "connecting you now", "please wait while i connect you",
    "press 1 to", "press 2 to", "to connect your call", "your call is important",
    "this call may be recorded", "this is an automated call", "main menu",
    "customer service", "representative", "stay on the line", "please hold",
    "enter your", "select from", "choose from", "say your selection",
    "if you know your party's extension", "to connect your call please",
    "press 1 to connect",
    "press 1 for more options", "press 2 for more options", "press pound",
    "press hash", "press the pound key", "press the hash key",
    "press 1", "press 2", "press 3", "press 4", "press 5",
    "1 for more options", "for more options", "or press 1 for more",
    "or press 1 for", "or press 1", "or press pound for further",
    "or press pound for", "or press pound", "press pound for further options",
    "press pound for further", "pound for further options", "further options",
    "more options please press", "more options","Drop a message"

    # VOICEMAIL COMMANDS
    "leave a message", "record your message",
    "at the tone", "after the beep",
    "hang up", "press pound", "press 1",
    "press 2", "press 3",

    # IVR NAVIGATION
    "main menu", "for more options",
    "press 1 for", "press 2 for",
    "to continue", "to proceed",

    # UNAVAILABILITY
    "not available", "unavailable",
    "can't take", "unable to",
    "sorry i missed", "we missed",

    # SYSTEM STATUS
    "mailbox full", "mailbox is full",
    "not set up", "has not been",
    "cannot accept", "system error",

    # CALLBACK
    "return your call", "call you back",
    "get back", "as soon as",
    "soon as possible",

    # GREETINGS/FAREWELLS
    "you've reached", "thank you for calling",
    "have a nice day", "goodbye",
    "thank you goodbye",

    # INSTRUCTIONS
    "please enter", "please say",
    "please speak", "using your",
    "followed by", "then press",

    # AUTOMATED
    "automated system", "ivr system",
    "call may be recorded",
    "for quality assurance",

    # TRANSFERS
    "transferring", "connecting",
    "please hold", "one moment",
    "your call is important",

    # TIME REFERENCES
    "business hours", "office hours",
    "currently closed", "after hours",

    # BINARY
    "press 1 to accept", "press 2 to decline",
    "say yes", "say no",

    # LEGAL/COMPLIANCE
    "for security", "for verification",
    "to protect", "to ensure",

    # MARKETING
    "special offer", "limited time",
    "act now", "don't miss",

    # QUEUE
    "estimated wait", "you are number",
    "in the queue", "all agents busy",

    # PERFECT PHRASES
    "if you know your party's",
    "please stay on the line",
    "to speak with a representative",

    # ROBOTIC
    "now transferring", "now connecting",
    "redirecting now", "please do not hang"," please record your message"

    # UNMODALIZED
    "please leave", "please record",
    "start speaking", "begin recording",
    "hello leave a message", "hello leave your message", "hello leave me a message",
    "hello leave a brief message", "hello leave your name", "hello leave your number",
    "hello leave your name and number", "hello leave your name and phone number",
    "hello please leave", "hello please record", "hello record your message",
    "hello start speaking", "hello begin recording", "hello message recorded", "hello message saved",
    "hello please record your message", "hello at the tone please record",
    "hello tone please record your", "hello tone please record",
    "hello record your message when", "hello when you've finished recording",
    "hello when you have finished recording", "hello when you are finished recording",
    "hello finished recording you may", "hello recording you may hang",
    "hello you may hang up or press", "hello you may hang up or", "hello you may hang up",
    "hello may hang up or press", "hello hang up or press pound", "hello hang up or press 1",
    "hello hang up or press", "hello simply hang up or press", "hello simply hang up or",
    "hello simply hang up", "hello recording simply hang", "hello please leave your name and",
    "hello please leave a message", "hello leave your name and", "hello leave a message and",
    "hello please leave your message after", "hello please leave a brief message",
    "hello please leave a message after", "hello please leave me a message",
    "hello leave your name and phone number", "hello leave your name and a",
    "hello leave your message after the", "hello leave your message after",
    "hello leave a message after the", "hello leave a message after",
    "hello leave me a message and", "hello your message when you", "hello your message when",
    "hello message when you've", "hello message when you have", "hello message when you",
    "hello finished recording", "hello when you have finished", "hello when you've finished",
    "hello you have finished", "hello have finished recording", "hello up or press", "hello or press","hello return the call thank you",
    "hello please press 1 to connect your call",

    # TONE & BEEP INDICATORS
    "hello at the tone", "hello after the tone", "hello after the beep", "hello beep",
    "hello at the tone please", "hello after the tone please", "hello message after the tone",
    "hello message after the beep", "hello the beep we will", "hello the beep",
    "hello the tone please", "hello the tone", "hello at the sound of the tone", "hello at the sound of the beep",
    "hello please speak after the tone", "hello please speak after the beep",
    "hello record a message at the tone", "hello record a message at the beep",
    "hello when you are done recording", "hello when you are done",
    "hello end of message", "hello to end your recording", "hello to end recording",
    "hello to finish recording", "hello to complete your message",
    "hello your message will be recorded", "hello your message will be saved",
    "hello to save your message", "hello to review your message",
    "hello press star to review", "hello press pound to save",
    "hello press any key to continue", "hello press any key",
    "hello wait for the prompt", "hello wait for the tone",
    "hello please wait for the tone", "hello please wait for the beep",
    "hello recording will begin after", "hello recording begins after",
    "hello you may begin speaking at the tone", "hello you may begin speaking at the beep",
    "hello begin speaking at the tone", "hello begin speaking at the beep",
    "hello start talking at the tone", "hello start talking at the beep",
    "hello at the sound", "hello after the sound", "hello sound of the tone",
    "hello sound of the beep", "hello you will hear a tone", "hello you will hear a beep",
    "hello listen for the tone", "hello listen for the beep",
    "hello tone will sound", "hello beep will sound", "hello short tone",
    "hello short beep", "hello long tone", "hello long beep","do you currently have both medicare Part A and part B", "leave a number",
    "do you have both parts A and B" , "can you confirm if you're covered under parts A and b?", "are you enrolled "
    ""

    # IVR NAVIGATION
    "hello press 0 for operator", "hello press 0 for assistance",
    "hello press 9 for directory", "hello press star for main menu",
    "hello press star to return", "hello press star to repeat",
    "hello to repeat this message", "hello to hear these options again",
    "hello for spanish press 2", "hello para español marque dos",
    "hello for other languages", "hello language options",
    "hello if this is an emergency", "hello in case of emergency",
    "hello to make a payment", "hello for billing inquiries",
    "hello for technical support", "hello for sales department",
    "hello to schedule an appointment", "hello to cancel an appointment",
    "hello to speak with a representative", "hello to speak with someone",
    "hello to reach a live person", "hello to talk to a person",

    # UNAVAILABILITY & STATUS
    "hello is not available", "hello not available", "hello unavailable", "hello not able to come",
    "hello unable to take your call", "hello unable to answer", "hello we are unable",
    "hello we are not", "hello can't take your call", "hello can't answer", "hello can't come",
    "hello can't get", "hello sorry i missed", "hello sorry i am not available",
    "hello i'm sorry i missed your call", "hello we missed your call", "hello forwarded",
    "hello nothing has been recorded", "hello is not available at the tone",
    "hello is not available at this time", "hello is not available at",
    "hello not available at the tone", "hello not available at the",
    "hello not available at this", "hello available at the tone",
    "hello is not available please", "hello can't take your call now at",
    "hello can't take your call now", "hello cannot take your call",
    "hello take your call now at the", "hello take your call now at", "hello take your call now",
    "hello currently unavailable", "hello temporarily unavailable",
    "hello out of the office", "hello away from desk", "hello away from phone",
    "hello on another call", "hello on the phone", "hello busy line",
    "hello line is busy", "hello please try again later", "hello please call back later",
    "hello office is closed", "hello office hours are", "hello business hours",
    "hello after hours", "hello holiday schedule", "hello vacation message",
    "hello out of town", "hello traveling", "hello on vacation", "hello on leave",
    "hello on sick leave", "hello medical leave", "hello maternity leave",
    "hello paternity leave", "hello will be back", "hello return on",

    # MAILBOX & SYSTEM MESSAGES
    "hello mailbox", "hello voice mailbox", "hello voicemail", "hello mailbox is full",
    "hello voicemail is full", "hello the mailbox is full", "hello cannot accept messages",
    "hello cannot accept any messages", "hello is full and cannot accept new messages",
    "hello voice messaging system", "hello has not been set up yet goodbye",
    "hello has not been set up yet", "hello not been set up yet", "hello not been set up",
    "hello mailbox is full and cannot accept", "hello mailbox is full and cannot",
    "hello is full and cannot accept new", "hello is full and cannot accept",
    "hello cannot accept new messages", "hello voice mailbox that has not",
    "hello voice message system", "hello try again later goodbye",
    "hello mailbox has not been initialized", "hello mailbox not configured",
    "hello voicemail not set up", "hello personal greeting not set",
    "hello system greeting", "hello default greeting", "hello generic greeting",
    "hello this number is not in service", "hello number disconnected",
    "hello number has been changed", "hello new number is",
    "hello this number is no longer in use", "hello out of service",
    "hello circuit busy", "hello all circuits are busy",
    "hello network busy", "hello system busy", "hello high call volume",
    "hello unexpected error", "hello system error", "hello technical difficulties",
    "hello please try your call again", "hello unable to complete your call",

    # CALLBACK PROMISES
    "hello return your call", "hello i will return your call", "hello we will return your call",
    "hello call you back", "hello i will call you back", "hello call me back",
    "hello get back to you", "hello i'll get back to you", "hello as soon as possible",
    "hello as soon as you can", "hello right now", "hello will return your call as soon",
    "hello return your call as soon", "hello i'll get back to you as soon",
    "hello get back to you as soon as", "hello get back to you as soon",
    "hello get back with you as soon", "hello get back with you",
    "hello call you back as soon", "hello as soon as possible thank you",
    "hello soon as possible thank", "hello soon as possible",
    "hello will get back to you", "hello shall return your call",
    "hello expect a callback", "hello expect a return call",
    "hello i'll try you back", "hello we'll try you back",
    "hello at my earliest convenience", "hello at our earliest convenience",
    "hello when i return", "hello when we return", "hello upon my return",
    "hello as soon as i can", "hello as soon as we can",
    "hello at the next opportunity", "hello next chance i get",
    "hello when possible", "hello when i'm available",

    # SCAM/ROBOCALL PATTERNS
    "hello verification required", "hello immediate attention required", "hello legal action",
    "hello support advisor", "hello unauthorized", "hello fraud", "hello medicare", "hello benefits",
    "hello qualify", "hello amazon", "hello shopify", "hello ebay", "hello to send a message",
    "hello to send an sms notification", "hello para continuar en español",
    "hello screened by smart call blocker", "hello smart call blocker",
    "hello say cancel or press", "hello your call",
    "hello urgent matter", "hello important information", "hello final notice",
    "hello account suspended", "hello account compromised", "hello security alert",
    "hello warranty expired", "hello warranty about to expire",
    "hello credit card offer", "hello loan approval", "hello debt consolidation",
    "hello student loan forgiveness", "hello social security",
    "hello irs", "hello internal revenue service", "hello tax debt",
    "hello free trial", "hello special offer", "hello limited time",
    "hello prize winner", "hello you have won", "hello congratulations you",
    "hello tech support", "hello microsoft support", "hello apple support",
    "hello windows support", "hello virus detected", "hello malware alert",
    "hello suspicious activity", "hello unusual login attempt",
    "hello press 1 to accept", "hello press 1 to claim",
    "hello press 1 to speak with", "hello press 1 now",
    "hello to be removed", "hello to unsubscribe",
    "hello do not press any keys", "hello do not hang up",

    # TELEMARKETING PHRASES
    "hello survey", "hello market research", "hello opinion poll",
    "hello political survey", "hello charity donation",
    "hello fundraising", "hello non-profit organization",
    "hello sales call", "hello product demonstration",
    "hello free estimate", "hello free consultation",
    "hello no obligation", "hello no cost",
    "hello exclusive offer", "hello pre-approved",

    # BUSINESS SPECIFIC
    "hello doctor's office", "hello medical practice",
    "hello dental office", "hello veterinary clinic",
    "hello law office", "hello attorney at law",
    "hello real estate office", "hello insurance agency",
    "hello financial advisor", "hello investment firm",
    "hello property management", "hello home services",
    "hello utility company", "hello cable company",
    "hello internet provider", "hello phone company",
    "hello bank", "hello credit union", "hello financial institution",

    # TIME AND DATE REFERENCES
    "hello office hours", "hello business hours", "hello open from",
    "hello closed on", "hello weekends", "hello holidays",
    "hello eastern time", "hello pacific time", "hello central time",
    "hello mountain time", "hello standard time", "hello daylight time",
    "hello today is", "hello current date", "hello current time",

    # SYSTEM PROMPTS
    "hello please enter", "hello please say", "hello please speak",
    "hello using your keypad", "hello using your voice",
    "hello followed by the pound key", "hello followed by pound",
    "hello then press pound", "hello then press hash",
    "hello to confirm press", "hello to verify press",
    "hello to continue press", "hello to proceed press",
    "hello to opt out", "hello to stop receiving calls",

    # CONNECTION MESSAGES
    "hello transferring", "hello connecting", "hello redirecting",
    "hello please hold while", "hello one moment please",
    "hello momentarily", "hello shortly", "hello briefly",
    "hello your call is important to us",
    "hello all agents are currently busy",
    "hello next available representative",

    # IVR GREETINGS
    "hello you've reached", "hello thank you for calling", "hello have a blessed day",
    "hello have a great day", "hello have a wonderful day", "hello have a good day",
    "hello thank you and have a", "hello you've reached the voicemail", "hello you have reached",
    "hello sorry i missed your call please", "hello sorry we missed your call",
    "hello i missed your call",
    "hello welcome to", "hello thank you for contacting",
    "hello thanks for calling", "hello appreciate your call",
    "hello good day", "hello take care", "hello talk to you soon",
    "hello looking forward to", "hello speak with you soon",
    "hello have a nice day", "hello have a productive day",
    "hello best regards", "hello warm regards", "hello sincerely",
    "hello respectfully", "hello yours truly",

    # IVR PHRASES
    "hello welcome to our automated system", "hello automated attendant",
    "hello interactive voice response", "hell o ivr system",
    "hello for quality assurance", "hello for training purposes",
    "hello calls are monitored", "hello calls are recorded",
    "hello your call will be answered", "hello in the order it was received",
    "hello estimated wait time", "hello current wait time",
    "hello you are number", "hello in the queue",
    "hello please remain on the line", "hello please do not hang up",
    "hello your call is being transferred", "hello transferring your call",
    "hello connecting you now", "hello please wait while i connect you",
    "hello press 1 to", "hello press 2 to", "hello to connect your call", "hello your call is important",
    "hello this call may be recorded", "hello this is an automated call", "hello main menu",
    "hello customer service", "hello representative", "hello stay on the line", "hello please hold",
    "hello enter your", "hello select from", "hello choose from", "hello say your selection",
    "hello if you know your party's extension", "hello to connect your call please",
    "hello press 1 to connect",
    "hello press 1 for more options", "hello press 2 for more options", "hello press pound",
    "hello press hash", "hello press the pound key", "hello press the hash key",
    "hello press 1", "hello press 2", "hello press 3", "hello press 4", "hello press 5",
    "hello press pound", "hello press hash", "hello press the pound key", "hello press the hash key",
    "hello 1 for more options", "hello for more options", "hello or press 1 for more",
    "hello or press 1 for", "hello or press 1", "hello or press pound for further",
    "hello or press pound for", "hello or press pound", "hello press pound for further options",
    "hello press pound for further", "hello pound for further options", "hello further options",
    "hello more options please press", "hello more options","phone ringing"
]


# AMD classifier

def classify_segment(text: str) -> str:
    """
    Returns 'machine' or 'human' for a given transcript block.

    Logic:
      - Count MACHINE keyword hits in the lowercased text.
      - Count HUMAN keyword hits in the lowercased text.
      - Machine wins on a tie (conservative default).
      - No match at all -> 'human' (live-call assumption).
    """
    t = text.lower()
    machine_hits = sum(1 for kw in MACHINE_KEYWORDS if kw in t)
    human_hits   = sum(1 for kw in HUMAN_KEYWORDS   if kw in t)

    if machine_hits == 0 and human_hits == 0:
        return "human"
    if machine_hits >= human_hits:
        return "machine"
    return "human"


# Helper functions

def build_model_name(model_size: str, language: str) -> str:
    name = model_size
    if language == "English" and model_size != "large":
        name += ".en"
    return name


def convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert any supported audio file to WAV via ffmpeg."""
    subprocess.call(
        ["ffmpeg", "-y", "-i", input_path, output_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def get_duration(wav_path: str) -> float:
    """Return duration in seconds of a WAV file."""
    with contextlib.closing(wave.open(wav_path, "r")) as f:
        return f.getnframes() / float(f.getframerate())


def segment_embedding(segment: dict, wav_path: str, duration: float,
                      audio_obj: Audio, embedding_model) -> np.ndarray:
    """Extract a 192-d ECAPA-TDNN speaker embedding for one Whisper segment."""
    start    = segment["start"]
    end      = min(duration, segment["end"])
    clip     = Segment(start, end)
    waveform, _ = audio_obj.crop(wav_path, clip)
    return embedding_model(waveform[None])


def assign_speakers(segments: list, embeddings: np.ndarray,
                    num_speakers: int) -> list:
    """
    Cluster embeddings and assign a SPEAKER label to each segment.

    Handles edge cases:
      - Only 1 segment -> skip clustering, label SPEAKER 1.
      - Fewer segments than num_speakers -> reduce cluster count accordingly.
    """
    embeddings = np.nan_to_num(embeddings)

    if len(segments) < 2:
        segments[0]["speaker"] = "SPEAKER 1"
        return segments

    effective_clusters = min(num_speakers, len(segments))
    clustering = AgglomerativeClustering(effective_clusters).fit(embeddings)
    for i, label in enumerate(clustering.labels_):
        segments[i]["speaker"] = f"SPEAKER {label + 1}"
    return segments


def fmt_time(secs: float) -> str:
    """Format seconds as H:MM:SS."""
    return str(datetime.timedelta(seconds=round(secs)))


def save_transcript(segments: list, audio_filename: str, combined_out_path: str) -> None:
    """
    Group consecutive same-speaker segments into blocks, classify each block
    as _human or _machine, then APPEND to the single combined transcript file.
    """
    blocks = []
    for seg in segments:
        if blocks and blocks[-1]["speaker"] == seg["speaker"]:
            blocks[-1]["text"] += seg["text"]
            blocks[-1]["end"]   = seg["end"]
        else:
            blocks.append({
                "speaker": seg["speaker"],
                "start":   seg["start"],
                "end":     seg["end"],
                "text":    seg["text"],
            })

    with open(combined_out_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"FILE: {audio_filename}\n")
        f.write(f"{'=' * 60}\n")

        for block in blocks:
            label     = classify_segment(block["text"])
            clean_txt = block["text"].strip()
            f.write(f"\n{block['speaker']} {fmt_time(block['start'])}\n")
            f.write(f"{clean_txt}_{label}\n")

    print(f"    Transcript appended -> {combined_out_path}")


# Process a single audio file

def process_file(audio_path: str, out_folder: str,
                 whisper_model, embedding_model,
                 audio_obj: Audio,
                 num_speakers: int,
                 combined_out_path: str) -> None:

    base_name      = os.path.splitext(os.path.basename(audio_path))[0]
    audio_filename = os.path.basename(audio_path)
    print(f"\n[Processing] {audio_filename}")

    # 1. Convert to WAV if needed
    needs_conversion = not audio_path.lower().endswith(".wav")
    if needs_conversion:
        wav_path = os.path.join(out_folder, base_name + "_temp.wav")
        print("    Converting to WAV ...")
        convert_to_wav(audio_path, wav_path)
        if not os.path.exists(wav_path):
            print("    Conversion failed - skipping.")
            return
    else:
        wav_path = audio_path

    # 2. Transcribe with Whisper
    print("    Transcribing ...")
    result   = whisper_model.transcribe(wav_path)
    segments = result["segments"]

    if not segments:
        print("    No speech detected - skipping.")
        if needs_conversion and os.path.exists(wav_path):
            os.remove(wav_path)
        return

    # 3. Build speaker embeddings
    print(f"    Building speaker embeddings ({len(segments)} segment(s)) ...")
    duration   = get_duration(wav_path)
    embeddings = np.zeros((len(segments), 192))
    for i, seg in enumerate(segments):
        try:
            embeddings[i] = segment_embedding(seg, wav_path, duration,
                                              audio_obj, embedding_model)
        except Exception:
            pass  # leave as zeros if a segment is too short / fails

    # 4. Cluster -> assign speaker labels
    segments = assign_speakers(segments, embeddings, num_speakers)

    # 5. Save transcript with AMD labels (appended to combined file)
    save_transcript(segments, audio_filename, combined_out_path)

    # 6. Clean up temporary WAV
    if needs_conversion and os.path.exists(wav_path):
        os.remove(wav_path)


# Main

def main():
    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Single combined output file path
    combined_out_path = os.path.join(OUTPUT_FOLDER, TRANSCRIPTION['output_file_name'])

    # Clear/create the combined file fresh at the start of each run
    with open(combined_out_path, "w", encoding="utf-8") as f:
        f.write("COMBINED TRANSCRIPTS\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        # Company line removed (was from config)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load Whisper
    model_name = build_model_name(MODEL_SIZE, LANGUAGE)
    print(f"Loading Whisper model '{model_name}' ...")
    whisper_mdl = whisper.load_model(model_name)

    # Load speaker embedding model
    print("Loading speaker embedding model ...")
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        device=device,
    )
    audio_obj = Audio()

    # Collect audio files
    audio_files = [
        os.path.join(AUDIO_FOLDER, f)
        for f in sorted(os.listdir(AUDIO_FOLDER))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ]

    if not audio_files:
        print(f"\nNo supported audio files found in: {AUDIO_FOLDER}")
        return

    print(f"\nFound {len(audio_files)} audio file(s) to process.\n")
    print(f"Combined transcript will be saved to: {combined_out_path}\n")

    # Process each file
    ok, failed = 0, 0
    for audio_path in audio_files:
        try:
            process_file(
                audio_path, OUTPUT_FOLDER,
                whisper_mdl, embedding_model, audio_obj,
                NUM_SPEAKERS,
                combined_out_path,
            )
            ok += 1
        except Exception as e:
            failed += 1
            print(f"    Error processing {os.path.basename(audio_path)}: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Done - {ok} succeeded, {failed} failed.")
    print(f"   Combined transcript saved to: {combined_out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()