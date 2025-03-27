#!/usr/bin/env python3
# This is like a magic computer program that helps find bad guys in computer logs!

# First, we get all our special tools ready (like getting crayons before drawing)
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import re
import argparse
from datetime import datetime

class LogAnalyzer:
    # This is like a detective that looks for clues in computer messages
    def __init__(self, model_path='model'):
        # We get our special detective tools ready
        self.model = load_model(f'{model_path}/detection_model.keras')
        self.scaler = joblib.load(f'{model_path}/scaler.pkl')
        self.feature_names = list(pd.read_csv('logs/web_features.csv').drop('label', axis=1).columns)

    def parse_log_line(self, log_line):
        # This is like reading a message and finding important parts
        # Like finding who wrote it, when they wrote it, and what they said
        log_pattern = r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\S+)\s+(\S+)(?:\[(\d+)\])?:\s+(.*)'
        match = re.match(log_pattern, log_line)
        if match:
            timestamp, host, program, pid, message = match.groups()
            return {'timestamp': timestamp, 'host': host, 'program': program,
                    'pid': pid if pid else '', 'message': message}
        return None

    def extract_features_single(self, log_entry):
        # This is like looking for special clues in the message
        # We check things like:
        # - What time it happened (like morning or night)
        # - If someone tried to use a wrong password
        # - If someone is trying to be sneaky
        # - If someone is trying to break things
        df = pd.DataFrame([log_entry])
        df['datetime'] = pd.to_datetime(df['timestamp'], format='%b %d %H:%M:%S', errors='coerce')
        df['hour'] = df['datetime'].dt.hour
        df['failed_password'] = df['message'].str.contains('Failed password', case=False).astype(int)
        df['accepted_login'] = df['message'].str.contains('Accepted', case=False).astype(int)
        ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        df['has_ip'] = df['message'].str.contains(ip_pattern).astype(int)
        ip_match = re.search(ip_pattern, log_entry['message'])
        ip_address = ip_match.group(0) if ip_match else None
        user_pattern = r'user (\w+)|for (\w+) from'
        user_match = re.search(user_pattern, log_entry['message'])
        username = user_match.group(1) or user_match.group(2) if user_match else None
        df['root_access'] = df['message'].str.contains('root', case=False).astype(int)
        df['user_mod'] = df['message'].str.contains('user|useradd|usermod|password changed', case=False).astype(int)
        df['sudo_cmd'] = df['message'].str.contains('sudo', case=False).astype(int)
        df['invalid_user'] = df['message'].str.contains('invalid user', case=False).astype(int)
        df['system_cmd'] = df['message'].str.contains('/bin/|/usr/bin/|wget|curl|bash', case=False).astype(int)
        df['suspicious_tool'] = df['message'].str.contains('nc |netcat|nmap', case=False).astype(int)
        df['log_manipulation'] = df['message'].str.contains('rm -rf|sed -i|/var/log', case=False).astype(int)
        df['backdoor'] = df['message'].str.contains('backdoor', case=False).astype(int)
        program_dummies = pd.DataFrame()
        for feature in self.feature_names:
            if feature.startswith('program_'):
                program_name = feature.replace('program_', '')
                program_dummies[feature] = [1 if df['program'].iloc[0] == program_name else 0]
        feature_df = pd.DataFrame(0, index=[0], columns=self.feature_names)
        for col in df.columns:
            if col in feature_df.columns:
                feature_df[col] = df[col].values
        for col in program_dummies.columns:
            if col in feature_df.columns:
                feature_df[col] = program_dummies[col].values
        return feature_df, ip_address, username

    def predict(self, log_line):
        # This is where we decide if something looks suspicious
        # Like how you know when your little brother is up to no good!
        # We look at all the clues and say "This looks okay" or "This looks bad!"
        log_entry = self.parse_log_line(log_line)
        if not log_entry: return None
        features, ip_address, username = self.extract_features_single(log_entry)
        scaled_features = self.scaler.transform(features)
        prediction = self.model.predict(scaled_features)[0][0]
        is_suspicious = bool(prediction > 0.5)
        return {'is_suspicious': is_suspicious, 'confidence': float(prediction),
                'ip_address': ip_address, 'username': username,
                'log_entry': log_entry, 'raw_log': log_line}

# This is the part that runs when we start our detective program
if __name__ == "__main__":
    # We ask what we should look at:
    # - Do you want us to look at one message?
    # - Or do you want us to look at lots of messages in a file?
    parser = argparse.ArgumentParser(description='Log file threat detection')
    parser.add_argument('--file', type=str, help='Path to the log file to analyze')
    parser.add_argument('--line', type=str, help='Single log line to analyze')
    args = parser.parse_args()

    # Get our detective ready!
    analyzer = LogAnalyzer()

    if args.line:
        # If we're looking at just one message
        # We check if it's suspicious and tell you what we found
        # Like saying "I found something fishy!"
        result = analyzer.predict(args.line)
        if result:
            print(f"Suspicious: {'Yes' if result['is_suspicious'] else 'No'}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"IP: {result['ip_address']}, Username: {result['username']}")

    elif args.file:
        # If we're looking at lots of messages in a file
        # We look at each one and count how many suspicious things we find
        # Like counting how many times someone did something naughty
        with open(args.file, 'r') as f:
            lines = f.readlines()
        suspicious_count = 0
        for line in lines:
            line = line.strip()
            result = analyzer.predict(line)
            if result and result['is_suspicious']:
                suspicious_count += 1
                print(f"SUSPICIOUS: {line}")
                print(f"Confidence: {result['confidence']:.4f}, IP: {result['ip_address']}")
        print(f"\nAnalysis complete. Found {suspicious_count} suspicious log entries out of {len(lines)} total.")
    else:
        # If we don't know what to look at, we ask for help
        print("Please provide a log file with --file or a single log line with --line")