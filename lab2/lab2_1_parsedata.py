#!/usr/bin/env python3
import pandas as pd
import os

def parse_firewall_log(file_path):
    # This is like opening a special book of computer stories and putting them in order
    # Just like how we organize toys in different boxes!
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading firewall log: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Make a new folder called "logs" to keep our computer stories safe
    # Like making a special drawer for your favorite things!
    os.makedirs("logs", exist_ok=True)

    # Let's read our computer story book
    firewall_logs = parse_firewall_log('@firewall_train_log.csv')

    # Now let's save our organized stories in a new book
    if not firewall_logs.empty:
        firewall_logs.to_csv('logs/processed_firewall_logs.csv', index=False)
        print(f"Processed {len(firewall_logs)} log entries. Saved to logs/processed_firewall_logs.csv")
    else:
        print("No data was processed. Please check the input file.")