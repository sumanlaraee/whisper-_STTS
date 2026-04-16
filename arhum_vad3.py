import os
import json
import time
import numpy as np
import joblib
import librosa
import glob
import tempfile
import subprocess
import warnings
from datetime import datetime
from scipy.signal import butter, sosfilt
import webrtcvad
from typing import Tuple, Optional, List, Dict
from faster_whisper import WhisperModel
import torch
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

ASTERISK_SR = 8000
WHISPER_SR = 16000

# Timing configuration
EARLY_SEC = float("2.0")
MAX_CALL_SEC = float("10.0")
EXTENDED_SEC = float("3.5")

# Whisper configuration
WHISPER_MODEL_DIR = "small.en"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "float32"

# VAD configuration
VAD_AGGRESSIVENESS = 2
MIN_VOICED_SPEECH_SEC = float("0.5")
VAD_FRAME_DURATION_MS = 30

# Audio thresholds
SILENCE_RMS_THRESHOLD = float("0.003")
ENERGY_RMS_THRESHOLD = float("0.008")

HIGHPASS_CUTOFF = 80
TARGET_RMS = float("0.1")
MAX_GAIN = float("3.0")

# Chunking settings
MIN_CHUNK_DURATION = 0.3
MAX_CHUNK_DURATION = 5.0

# Whisper prompt optimized for telephony/voicemail
INITIAL_PROMPT = (
    "Phone call greeting: Hello, hi, yes, speaking, good morning, good afternoon, "
    "this is, can you hear me, how may I help you, thank you for calling, "
    "please leave a message after the beep, the person you are trying to reach"
)

# Known Whisper hallucinations
HALLUCINATIONS = [
    "thanks for watching", "thank you for watching", "please subscribe",
    "welcome back", "this video", "like and subscribe", "hit the bell",
    "see you next time", "subtitles by", "captions by", "transcribed by",
    "[music]", "(music)", "[applause]", "[silence]", "[inaudible]",
]

# Keywords lists
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
    "leave a message", "after the beep", "after the tone","please drop a message.",
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
    "hello short beep", "hello long tone", "hello long beep",
    
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

print("=" * 78)
print(" AMD Detection - VAD + Confidence Scores Pipeline")
print(f" Device: {DEVICE}")
print(f" Whisper: {WHISPER_MODEL_DIR}")
print("=" * 78)

# Load Whisper model
whisper_model = WhisperModel(
    WHISPER_MODEL_DIR,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
)
print("✓ Whisper loaded")

# Initialize VAD
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
print("✓ WebRTC VAD ready")

# Pre-compute highpass filter coefficients
sos_8k = butter(4, HIGHPASS_CUTOFF, btype='high', fs=ASTERISK_SR, output='sos')
print("✓ Audio filter initialized")

# Suppress NNPACK warnings
os.environ['NNPACK_VERBOSE'] = '0'

# ==================== HELPER FUNCTIONS ====================

def detect_speech_with_vad(audio_16k: np.ndarray, sr: int = 16000) -> List[Dict]:
    """Detect speech segments using WebRTC VAD."""
    if len(audio_16k) == 0:
        return []
    
    frame_duration_ms = VAD_FRAME_DURATION_MS
    frame_samples = int(sr * frame_duration_ms / 1000)
    
    audio_int16 = (audio_16k * 32767).astype(np.int16)
    
    speech_segments = []
    in_speech = False
    speech_start = 0
    silence_frames = 0
    max_silence_frames = 6
    
    for i in range(0, len(audio_int16) - frame_samples, frame_samples):
        frame = audio_int16[i:i + frame_samples]
        
        try:
            is_speech = vad.is_speech(frame.tobytes(), sr)
        except:
            is_speech = False
        
        if is_speech:
            if not in_speech:
                in_speech = True
                speech_start = i / sr
            silence_frames = 0
        elif in_speech:
            silence_frames += 1
            if silence_frames >= max_silence_frames:
                in_speech = False
                speech_end = (i - silence_frames * frame_samples) / sr
                duration = speech_end - speech_start
                
                if duration >= MIN_CHUNK_DURATION:
                    speech_segments.append({
                        'start': round(speech_start, 2),
                        'end': round(speech_end, 2),
                        'duration': round(duration, 2)
                    })
    
    if in_speech:
        speech_end = len(audio_int16) / sr
        duration = speech_end - speech_start
        if duration >= MIN_CHUNK_DURATION:
            speech_segments.append({
                'start': round(speech_start, 2),
                'end': round(speech_end, 2),
                'duration': round(duration, 2)
            })
    
    return speech_segments

def chunk_segments_by_duration(segments: List[Dict], audio_16k: np.ndarray,
                               max_chunk_duration: float = MAX_CHUNK_DURATION,
                               sr: int = 16000) -> List[Dict]:
    """Further chunk VAD segments into smaller duration-based chunks."""
    all_chunks = []
    
    for seg in segments:
        seg_duration = seg['end'] - seg['start']
        
        if seg_duration <= max_chunk_duration:
            all_chunks.append({
                'start': seg['start'],
                'end': seg['end'],
                'duration': round(seg_duration, 2),
                'chunk_type': 'original'
            })
        else:
            num_chunks = int(np.ceil(seg_duration / max_chunk_duration))
            chunk_size = seg_duration / num_chunks
            
            for i in range(num_chunks):
                chunk_start = seg['start'] + i * chunk_size
                chunk_end = min(seg['start'] + (i + 1) * chunk_size, seg['end'])
                chunk_duration = chunk_end - chunk_start
                
                all_chunks.append({
                    'start': round(chunk_start, 2),
                    'end': round(chunk_end, 2),
                    'duration': round(chunk_duration, 2),
                    'chunk_type': 'subdivided'
                })
    
    return all_chunks

def remove_repeated_text(text: str) -> str:
    """Remove repeated phrases/words from transcription."""
    if not text or len(text.strip()) < 2:
        return text
    
    words = text.lower().split()
    if len(words) == 0:
        return text
    
    cleaned_words = []
    prev_word = None
    
    for word in words:
        if word != prev_word:
            cleaned_words.append(word)
            prev_word = word
    
    phrase_positions = {}
    for i in range(len(cleaned_words) - 1):
        phrase = f"{cleaned_words[i]} {cleaned_words[i+1]}"
        if phrase not in phrase_positions:
            phrase_positions[phrase] = []
        phrase_positions[phrase].append(i)
    
    skip_indices = set()
    for phrase, positions in phrase_positions.items():
        if len(positions) > 2:
            for pos in positions[2:]:
                skip_indices.add(pos)
                skip_indices.add(pos + 1)
    
    result_words = [cleaned_words[i] for i in range(len(cleaned_words)) if i not in skip_indices]
    cleaned_text = " ".join(result_words)
    
    return cleaned_text

def get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        duration = librosa.get_duration(filename=file_path)
        return float(duration)
    except:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                 "-of", "default=noprint_wrappers=1:nokey=1", file_path],
                capture_output=True, text=True, timeout=10
            )
            return float(result.stdout.strip())
        except:
            return 0.0

def load_audio_file(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and convert to target sample rate."""
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        return audio.astype(np.float32)
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([], dtype=np.float32)

def rms(audio: np.ndarray) -> float:
    """Calculate root mean square of audio."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio ** 2)))

def balanced_normalize(audio: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    """Balanced normalization with moderate gain."""
    if len(audio) == 0:
        return audio
    
    current_rms = rms(audio)
    if current_rms < 1e-6:
        return audio
    
    gain = target_rms / current_rms
    gain = min(gain, MAX_GAIN)
    
    amplified = audio * gain
    clipped = np.clip(amplified, -1.0, 1.0)
    
    return clipped.astype(np.float32)

def highpass_filter(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply highpass filter to remove low-frequency noise."""
    if len(audio) < 100:
        return audio
    
    sos = butter(4, HIGHPASS_CUTOFF, btype='high', fs=sr, output='sos')
    return sosfilt(sos, audio).astype(np.float32)

def preprocess_audio_for_whisper(audio_16k: np.ndarray) -> np.ndarray:
    """Preprocess audio for Whisper transcription."""
    if len(audio_16k) == 0:
        return audio_16k
    
    filtered = highpass_filter(audio_16k, 16000)
    normalized = balanced_normalize(filtered)
    
    return normalized

def is_hallucination(text: str) -> bool:
    """Check if transcription is a known Whisper hallucination."""
    if not text:
        return True
    
    t = text.lower().strip()
    
    if len(t) < 2:
        return True
    
    for h in HALLUCINATIONS:
        if h in t:
            return True
    
    clean = t.replace(" ", "")
    if len(clean) > 3 and len(set(clean)) <= 1:
        return True
    
    words = t.split()
    if len(words) >= 4:
        unique_words = set(words)
        if len(unique_words) == 1:
            return True
    
    return False

def extract_audio_segment_from_file(file_path: str, start_time: float, end_time: float) -> np.ndarray:
    """Extract audio segment directly from file using ffmpeg."""
    try:
        tmp_seg = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_seg_path = tmp_seg.name
        tmp_seg.close()
        
        cmd = [
            "ffmpeg", "-y", "-i", file_path,
            "-ss", str(start_time), "-to", str(end_time),
            "-ac", "1", "-ar", "16000", "-vn",
            "-acodec", "pcm_s16le", tmp_seg_path
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        audio = load_audio_file(tmp_seg_path, target_sr=16000)
        os.unlink(tmp_seg_path)
        
        return audio
        
    except Exception as e:
        print(f"Error extracting segment: {e}")
        return np.array([], dtype=np.float32)

# ==================== CONFIDENCE SCORING FUNCTIONS ====================

def calculate_combined_confidence(whisper_conf: float, 
                                  keyword_conf: float,
                                  text_length: int,
                                  method: str) -> float:
    """Calculate combined confidence score from multiple factors."""
    if "keyword" in method:
        base_conf = keyword_conf
        length_boost = min(0.1, text_length * 0.02)
        return min(1.0, base_conf + length_boost)
    
    elif "heuristic" in method:
        return (whisper_conf * 0.7 + min(1.0, text_length / 20) * 0.3)
    
    else:
        return whisper_conf

def keyword_based_detection_enhanced(text: str) -> Tuple[Optional[str], float, str, float]:
    """Enhanced keyword-based detection with confidence breakdown."""
    if not text or len(text.strip()) < 2:
        return None, 0.0, "", 0.0
    
    text_lower = text.lower().strip()
    
    # Check MACHINE keywords first
    for keyword in MACHINE_KEYWORDS:
        if keyword in text_lower:
            keyword_words = len(keyword.split())
            if keyword_words >= 3:
                confidence = 0.95
            elif keyword_words == 2:
                confidence = 0.90
            else:
                confidence = 0.85
            return "MACHINE", confidence, keyword, confidence
    
    # Then check HUMAN keywords
    for keyword in HUMAN_KEYWORDS:
        if keyword in text_lower:
            keyword_words = len(keyword.split())
            confidence = 0.90 if keyword_words > 1 else 0.80
            return "HUMAN", confidence, keyword, confidence
    
    return None, 0.0, "", 0.0

def classify_text_hybrid_enhanced(text: str) -> Tuple[str, float, str, dict]:
    """Enhanced hybrid classification with detailed confidence breakdown."""
    if not text or len(text.strip()) < 2:
        return "HUMAN", 0.6, "empty_text", {"empty": 0.6}
    
    keyword_decision, keyword_conf, matched_keyword, keyword_only_conf = keyword_based_detection_enhanced(text)
    
    if keyword_decision is not None:
        method = f"keyword:'{matched_keyword}'"
        details = {
            "keyword_confidence": keyword_only_conf,
            "matched_keyword": matched_keyword,
            "text_length_words": len(text.split())
        }
        return keyword_decision, keyword_conf, method, details
    
    text_lower = text.lower().strip()
    word_count = len(text_lower.split())
    
    if word_count > 15:
        confidence = 0.70
        details = {"heuristic_type": "long_text", "word_count": word_count}
        return "MACHINE", confidence, "heuristic:long_text", details
    
    if word_count <= 3:
        confidence = 0.75
        details = {"heuristic_type": "short_text", "word_count": word_count}
        return "HUMAN", confidence, "heuristic:short_text", details
    
    confidence = 0.60
    details = {"heuristic_type": "default", "word_count": word_count}
    return "HUMAN", confidence, "heuristic:default", details

def transcribe_segment_enhanced(audio_16k: np.ndarray) -> Tuple[str, float, dict]:
    """Enhanced transcription with detailed confidence breakdown."""
    if len(audio_16k) < int(0.3 * WHISPER_SR):
        return "", 0.0, {"reason": "too_short"}
    
    try:
        segments, info = whisper_model.transcribe(
            audio_16k,
            language="en",
            beam_size=3,
            best_of=3,
            temperature=[0.0, 0.2],
            initial_prompt=INITIAL_PROMPT,
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=True,
            compression_ratio_threshold=2.0,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        
        texts = []
        confidences = []
        word_counts = []
        
        for seg in segments:
            text = seg.text.strip()
            if text and not is_hallucination(text):
                texts.append(text)
                word_counts.append(len(text.split()))
                
                if hasattr(seg, 'avg_logprob'):
                    conf = min(1.0, max(0.0, 1.0 + seg.avg_logprob))
                    confidences.append(conf)
                else:
                    confidences.append(0.8)
        
        full_text = " ".join(texts).strip()
        
        if full_text:
            full_text = remove_repeated_text(full_text)
        
        if is_hallucination(full_text):
            return "", 0.0, {"reason": "hallucination_detected"}
        
        avg_conf = np.mean(confidences) if confidences else 0.0
        
        confidence_details = {
            "whisper_avg_confidence": round(avg_conf, 3),
            "whisper_min_confidence": round(min(confidences), 3) if confidences else 0.0,
            "whisper_max_confidence": round(max(confidences), 3) if confidences else 0.0,
            "segments_transcribed": len(texts),
            "total_words": sum(word_counts)
        }
        
        return full_text, avg_conf, confidence_details
        
    except Exception as e:
        print(f"[TRANSCRIBE ERROR] {type(e).__name__}: {e}")
        return "", 0.0, {"reason": f"error: {str(e)}"}

# ==================== MAIN PROCESSING FUNCTIONS ====================

def process_audio_file_enhanced(file_path: str, output_file: str, json_output_file: str) -> Dict:
    """Process audio with enhanced confidence tracking."""
    print(f"\n{'='*80}")
    print(f"Processing: {os.path.basename(file_path)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    duration = get_audio_duration(file_path)
    print(f"Duration: {duration:.2f} seconds")
    
    # STEP 1: LOAD AUDIO
    print("Step 1: Loading audio...")
    audio_16k = load_audio_file(file_path, target_sr=16000)
    if len(audio_16k) == 0:
        print("❌ Failed to load audio file")
        return {}
    
    # STEP 2: VAD DETECTION
    print("Step 2: Detecting speech segments with VAD...")
    vad_start = time.time()
    vad_segments = detect_speech_with_vad(audio_16k, sr=16000)
    vad_time = time.time() - vad_start
    
    if not vad_segments:
        print("⚠️  No speech detected")
        return {}
    
    print(f"Found {len(vad_segments)} speech segments in {vad_time:.2f}s")
    
    # STEP 3: DURATION-BASED CHUNKING
    print("Step 3: Chunking segments by duration...")
    chunk_start = time.time()
    all_chunks = chunk_segments_by_duration(vad_segments, audio_16k, max_chunk_duration=MAX_CHUNK_DURATION)
    chunk_time = time.time() - chunk_start
    
    print(f"Created {len(all_chunks)} chunks in {chunk_time:.2f}s")
    
    # STEP 4: TRANSCRIBE & CLASSIFY with enhanced confidence
    print("Step 4: Transcribing and classifying chunks...")
    results = []
    
    for i, chunk in enumerate(all_chunks):
        segment_audio = extract_audio_segment_from_file(file_path, chunk['start'], chunk['end'])
        
        if len(segment_audio) == 0:
            continue
        
        processed_audio = preprocess_audio_for_whisper(segment_audio)
        
        transcribe_start = time.time()
        text, whisper_conf, whisper_details = transcribe_segment_enhanced(processed_audio)
        transcribe_time = time.time() - transcribe_start
        
        if text:
            decision, confidence, method, classify_details = classify_text_hybrid_enhanced(text)
            
            combined_confidence = calculate_combined_confidence(
                whisper_conf=whisper_conf,
                keyword_conf=classify_details.get("keyword_confidence", whisper_conf),
                text_length=len(text.split()),
                method=method
            )
            
            results.append({
                'segment_id': i,
                'start': chunk['start'],
                'end': chunk['end'],
                'text': text,
                'decision': decision,
                'confidence': round(combined_confidence, 3),
                'method': method,
                'whisper_confidence': round(whisper_conf, 3),
                'transcribe_time': transcribe_time,
                'chunk_type': chunk.get('chunk_type', 'original')
            })
            
            display_text = text[:50] + "..." if len(text) > 50 else text
            print(f"  Chunk {i}: {display_text} -> {decision} (conf: {combined_confidence:.2f})")
        else:
            print(f"  Chunk {i}: ({chunk['start']:.2f}s-{chunk['end']:.2f}s): No speech")
    
    # COMPILE RESULTS
    file_results = {
        'filename': os.path.basename(file_path),
        'full_path': file_path,
        'duration': duration,
        'vad_segment_count': len(vad_segments),
        'chunk_count': len(results),
        'segments': results,
        'processing_time': vad_time + chunk_time + sum(r.get('transcribe_time', 0) for r in results),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    save_results_to_file_enhanced(file_results, output_file, json_output_file)
    
    return file_results

def save_results_to_file_enhanced(results: Dict, output_file: str, json_output_file: str):
    """Save results with confidence score details to both TXT and JSON."""
    try:
        # Save TXT file (compact format)
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"FILE: {results['filename']}\n")
            f.write("="*80 + "\n")
            f.write(f"DURATION: {results['duration']:.2f}s | VAD SEGMENTS: {results['vad_segment_count']} | CHUNKS: {results['chunk_count']}\n")
            
            for seg in results['segments']:
                start_str = f"{seg['start']:7.2f}s"
                end_str = f"{seg['end']:7.2f}s"
                text = seg['text'].strip()
                label = seg['decision'].lower()
                conf = seg['confidence']
                
                f.write(f"[{start_str} → {end_str}]: {text} _{label} (conf: {conf:.3f})\n")
            
            f.write(f"Processing time: {results['processing_time']:.2f}s\n")
            f.write("="*80 + "\n")
            f.flush()
        
        print(f"✓ TXT results saved")
        
        # Save JSON file
        json_results = {
            'filename': results['filename'],
            'full_path': results['full_path'],
            'duration': round(results['duration'], 2),
            'vad_segment_count': results['vad_segment_count'],
            'chunk_count': results['chunk_count'],
            'timestamp': results['timestamp'],
            'segments': []
        }
        
        for seg in results['segments']:
            json_seg = {
                'segment_id': seg['segment_id'],
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'],
                'decision': seg['decision'],
                'confidence': round(seg['confidence'], 3),
                'method': seg['method'],
                'whisper_confidence': round(seg.get('whisper_confidence', 0), 3),
                'transcribe_time': round(seg['transcribe_time'], 3),
                'chunk_type': seg['chunk_type']
            }
            json_results['segments'].append(json_seg)
        
        json_results['processing_time'] = round(results['processing_time'], 2)
        
        # Load existing JSON or create new
        if os.path.exists(json_output_file):
            with open(json_output_file, 'r', encoding='utf-8') as f:
                try:
                    all_results = json.load(f)
                except:
                    all_results = []
        else:
            all_results = []
        
        all_results.append(json_results)
        
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ JSON results saved to {json_output_file}")
        
    except Exception as e:
        print(f"Error saving results: {e}")

def print_results(results: Dict):
    """Print results to console."""
    print(f"\n{'='*80}")
    print(f"FILE: {results['filename']}")
    print(f"{'='*80}")
    print(f"DURATION: {results['duration']:.2f}s | VAD SEGMENTS: {results['vad_segment_count']} | CHUNKS: {results['chunk_count']}\n")
    
    for seg in results['segments']:
        start_str = f"{seg['start']:7.2f}s"
        end_str = f"{seg['end']:7.2f}s"
        text = seg['text'].strip()
        label = seg['decision'].lower()
        conf = seg['confidence']
        
        print(f"[{start_str} → {end_str}]: {text} _{label} (conf: {conf:.3f})")
    
    print(f"\nProcessing time: {results['processing_time']:.2f}s")
    print(f"{'='*80}")

# ==================== MAIN EXECUTION ====================

def main():
    """Main function to process audio files."""
    audio_folder = "data/recordings_all5"
    
    if not os.path.exists(audio_folder):
        print(f"ERROR: Folder '{audio_folder}' does not exist!")
        return
    
    audio_files = []
    extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.ogg']
    
    for ext in extensions:
        audio_files.extend(glob.glob(os.path.join(audio_folder, f"**/{ext}"), recursive=True))
    
    print(f"Found {len(audio_files)} audio files")
    
    if len(audio_files) == 0:
        print("No audio files found!")
        return
    
    audio_files.sort()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file_name = f"AMD_RESULTS_VAD_{timestamp}.txt"
    json_output_file = f"AMD_RESULTS_VAD_{timestamp}.json"
    
    with open(output_file_name, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("AMD DETECTION - VAD + CONFIDENCE SCORES PIPELINE\n")
        f.write("Pipeline: Load => VAD => Duration Chunks => Whisper + Cleaning => Classification\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total files: {len(audio_files)}\n")
        f.write("="*80 + "\n\n")
    
    print(f"\n✓ Output files: {output_file_name} & {json_output_file}\n")
    
    print(f"You have {len(audio_files)} files to process.")
    response = input("How many files to process? (Enter number or 'all'): ").strip()
    
    if response.lower() == 'all':
        files_to_process = audio_files
    else:
        try:
            num_files = int(response)
            files_to_process = audio_files[:num_files]
        except:
            print("Invalid input. Processing first 5 files.")
            files_to_process = audio_files[:5]
    
    all_results = []
    start_time = time.time()
    
    for i, audio_file in enumerate(files_to_process):
        print(f"\n{'#'*80}")
        print(f"File {i+1}/{len(files_to_process)}: {os.path.basename(audio_file)}")
        print(f"{'#'*80}")
        
        results = process_audio_file_enhanced(audio_file, output_file_name, json_output_file)
        
        if results:
            print_results(results)
            all_results.append(results)
        
        if i < len(files_to_process) - 1:
            time.sleep(0.2)
    
    # SUMMARY
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    
    total_chunks = sum(r.get('chunk_count', 0) for r in all_results)
    human_chunks = sum(1 for r in all_results for seg in r.get('segments', []) if seg['decision'] == 'HUMAN')
    machine_chunks = sum(1 for r in all_results for seg in r.get('segments', []) if seg['decision'] == 'MACHINE')
    
    print(f"Files processed: {len(all_results)}")
    print(f"Total chunks: {total_chunks}")
    print(f"HUMAN: {human_chunks} | MACHINE: {machine_chunks}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"\n✓ Results saved in:")
    print(f"  - {output_file_name}")
    print(f"  - {json_output_file}")
    
    with open(output_file_name, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write("FINAL SUMMARY\n")
        f.write(f"{'='*80}\n")
        f.write(f"Files processed: {len(all_results)}\n")
        f.write(f"Total chunks: {total_chunks}\n")
        f.write(f"HUMAN chunks: {human_chunks}\n")
        f.write(f"MACHINE chunks: {machine_chunks}\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("AMD Detection - Optimized VAD Pipeline with Confidence Scores")
    print("="*80 + "\n")
    
    main()
