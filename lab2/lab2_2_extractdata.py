#!/usr/bin/env python3
import pandas as pd
import numpy as np
import re
import os

def extract_features(df):
    # Extract meaningful features from firewall log entries for ML processing
    
    # Extract time if timestamp exists
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['hour'] = df['datetime'].dt.hour
    
    # Extract IP addresses
    ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    df['has_ip'] = df['message'].str.contains(ip_pattern).astype(int)
    
    # Firewall specific features
    df['blocked'] = df['message'].str.contains('block|deny|reject', case=False).astype(int)
    df['allowed'] = df['message'].str.contains('allow|accept|permit', case=False).astype(int)
    
    # Port related features
    df['high_risk_ports'] = df['message'].str.contains('port (?:21|22|23|445|3389)', case=False).astype(int)
    
    # Protocol features
    df['tcp'] = df['message'].str.contains('TCP|tcp', case=False).astype(int)
    df['udp'] = df['message'].str.contains('UDP|udp', case=False).astype(int)
    df['icmp'] = df['message'].str.contains('ICMP|icmp', case=False).astype(int)
    
    # Scan detection
    df['potential_scan'] = df['message'].str.contains('scan|probe|sweep', case=False).astype(int)
    
    # DOS attempt detection
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
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)

    # Load firewall logs with proper column names
    column_names = ['timestamp', 'source_ip', 'dest_ip', 'protocol', 'source_port', 'dest_port', 'action']
    logs_df = pd.read_csv('logs/processed_firewall_logs.csv', names=column_names)

    # Convert action to message format for feature extraction
    logs_df['message'] = logs_df.apply(lambda row: f"{row['action']} {row['protocol']} connection from {row['source_ip']} to {row['dest_ip']} port {row['dest_port']}", axis=1)

    # Extract features
    features_df = extract_features(logs_df)

    # Add label column (1 for DENY, 0 for ALLOW)
    features_df['label'] = (logs_df['action'] == 'DENY').astype(int)

    # Save features
    features_df.to_csv('logs/firewall_features.csv', index=False)

    # Save feature names for later use
    with open('model/firewall_feature_names.txt', 'w') as f:
        f.write('\n'.join(features_df.columns[:-1] if 'label' in features_df.columns else features_df.columns))

    print(f"Extracted {features_df.shape[1]} features from {len(features_df)} firewall log entries.")
    print(f"Features saved to logs/firewall_features.csv")
    print(f"Feature names saved to model/firewall_feature_names.txt")