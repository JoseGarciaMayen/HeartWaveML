import os
import re

import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.signal import butter, filtfilt

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")


def get_record_numbers():
    """
    Returns a list of record numbers from the MIT-BIH Arrhythmia Database.
    """
    database_path = "data/raw/mit-bih-arrhythmia-database-1.0.0"

    record_numbers = []
    for filename in os.listdir(database_path):
        if filename.endswith(".atr"):
            match = re.match(r"^(\d+)\.atr$", filename)
            if match:
                record_numbers.append(match.group(1))

    return record_numbers


def get_class_mapping():
    """
    Returns a mapping of class labels to their corresponding integer values.
    """
    class_mapping = {
        "N": 0,
        "·": 0,
        "L": 0,
        "R": 0,
        "e": 0,
        "j": 0,
        "A": 1,
        "a": 1,
        "J": 1,
        "S": 1,
        "V": 2,
        "E": 2,
        "F": 3,
        "/": 4,
        "f": 4,
        "x": 4,
        "Q": 4,
        "|": 4,
        "~": 4,
    }
    return class_mapping


def get_filter_coeffs(cutoff_freq=40, fs=360, order=5):
    """
    Designs a Butterworth low-pass filter and returns its coefficients.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def apply_filter(data, b, a):
    """
    Applies a Butterworth filter to the given data.
    """
    return filtfilt(b, a, data)


def get_class_weights():
    """
    Computes class weights to handle class imbalance in the dataset.
    """
    train_df = pd.read_csv("data/processed/feat/mitbih_train_features.csv")
    y_train = train_df["class"]

    class_weights = {}
    unique_classes = sorted(y_train.unique())

    for cls in unique_classes:
        class_weights[cls] = len(y_train) / (len(unique_classes) * sum(y_train == cls))

    return class_weights


def notify_telegram(msg):
    """
    Sends a notification message to a specified Telegram chat.

    Credentials are optional: if they are missing or the request fails, the
    error is logged and swallowed so it never interrupts the caller (e.g. a
    long training run).
    """
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("Telegram notification skipped: TELEGRAM_TOKEN or CHAT_ID not set.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(url, data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Telegram notification failed: {e}")
