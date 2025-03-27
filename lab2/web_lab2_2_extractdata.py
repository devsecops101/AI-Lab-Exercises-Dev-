#!/usr/bin/env python3
import pandas as pd
import os

# This is a special program that helps us understand computer security logs!
# Just like how we look for clues in a detective game 

def extract_features(df):
    # We're going to look for special patterns in our computer messages
    # Like finding specific pieces in a puzzle! 
    
    # First, we check what time things happened
    # Just like remembering when you had lunch or dinner! 
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = df['datetime'].dt.hour
    
    # Look for computer addresses (like house addresses, but for computers!) 
    ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    df['has_ip'] = df['message'].str.contains(ip_pattern).astype(int)
    
    # Check if the computer said "no" to someone trying to enter
    # Like when mom says you can't have more cookies! 
    df['blocked'] = df['message'].str.contains('block|deny|reject', case=False).astype(int)
    df['allowed'] = df['message'].str.contains('allow|accept|permit', case=False).astype(int)
    
    # Look for special door numbers that might be dangerous
    # Like checking if someone is trying to enter through the window instead of the front door! 
    df['high_risk_ports'] = df['message'].str.contains('port (?:21|22|23|445|3389)', case=False).astype(int)
    
    # Check what kind of way computers are talking to each other
    # Like how we can talk, whisper, or shout! 
    df['tcp'] = df['message'].str.contains('TCP|tcp', case=False).astype(int)
    df['udp'] = df['message'].str.contains('UDP|udp', case=False).astype(int)
    df['icmp'] = df['message'].str.contains('ICMP|icmp', case=False).astype(int)
    
    # Check if someone is trying to peek around too much
    # Like when someone is being too nosy! 
    df['potential_scan'] = df['message'].str.contains('scan|probe|sweep', case=False).astype(int)
    
    # Look for anyone trying to send too many messages at once
    # Like when someone keeps pressing the doorbell over and over! 
    df['dos_attempt'] = df['message'].str.contains('flood|dos|ddos|overflow', case=False).astype(int)
    
    # If program column exists, create dummies
    if 'program' in df.columns:
        program_dummies = pd.get_dummies(df['program'], prefix='program')
        
        # Combine all features
        features = pd.concat([
            df[['hour', 'has_ip', 'blocked', 'allowed', 'high_risk_ports',
                'tcp', 'udp', 'icmp', 'potential_scan', 'dos_attempt']],
            program_dummies
        ], axis=1)
    else:
        features = df[['has_ip', 'blocked', 'allowed', 'high_risk_ports',
                      'tcp', 'udp', 'icmp', 'potential_scan', 'dos_attempt']]

    return features

if __name__ == "__main__":
    # Make a special folder to keep our findings
    # Like having a toy box to store all our toys! 
    os.makedirs("model", exist_ok=True)

    # Read our special computer diary
    # Like reading a storybook about what happened! 
    column_names = ['timestamp', 'source_ip', 'dest_ip', 'protocol', 'source_port', 'dest_port', 'action']
    logs_df = pd.read_csv('logs/processed_firewall_logs.csv', names=column_names)

    # Make the computer messages easier to understand
    # Like turning big words into a simple story! 
    logs_df['message'] = logs_df.apply(lambda row: f"{row['action']} {row['protocol']} connection from {row['source_ip']} to {row['dest_ip']} port {row['dest_port']}", axis=1)

    # Look for all our special clues
    # Like collecting all the pieces of a treasure map! 
    features_df = extract_features(logs_df)

    # Mark which messages were saying "no"
    # Like putting a red sticker on things that aren't allowed! 
    features_df['label'] = (logs_df['action'] == 'DENY').astype(int)

    # Save all our findings in a special file
    # Like putting our drawing in a safe place! 
    features_df.to_csv('logs/firewall_features.csv', index=False)

    # Write down the names of all the clues we found
    # Like making a list of all our favorite toys! 
    with open('model/firewall_feature_names.txt', 'w') as f:
        f.write('\n'.join(features_df.columns[:-1] if 'label' in features_df.columns else features_df.columns))

    # Tell everyone what we found
    # Like showing mom and dad what we did today! 
    print(f"Extracted {features_df.shape[1]} features from {len(features_df)} firewall log entries.")
    print(f"Features saved to logs/firewall_features.csv")
    print(f"Feature names saved to model/firewall_feature_names.txt")