#!/usr/bin/env python3
import pandas as pd
import os
import re

def parse_web_log(file_path):
    # This is like a special magic spell that helps us find all the important parts in our computer story!
    pattern = r'(\d+\.\d+\.\d+\.\d+) - - \[(.*?)\] "(.*?) (.*?) HTTP/\d\.\d" (\d+) (\d+) "(.*?)" "(.*?)"'
    
    # We're making a big empty basket to collect all our computer stories
    log_entries = []
    try:
        # Let's open our special storybook and read it line by line
        with open(file_path, 'r') as file:
            for line in file:
                match = re.match(pattern, line)
                if match:
                    # Here we're taking each story and breaking it into tiny pieces, like sorting toys!
                    ip, timestamp, method, path, status, bytes_sent, referrer, user_agent = match.groups()
                    # Now we put each piece in its own special box with a nice label
                    log_entries.append({
                        'ip_address': ip,
                        'timestamp': timestamp,
                        'method': method,
                        'path': path,
                        'status_code': int(status),
                        'bytes_sent': int(bytes_sent),
                        'referrer': referrer,
                        'user_agent': user_agent
                    })
        
        # Now we're putting all our sorted toys into one big toy box!
        df = pd.DataFrame(log_entries)
        return df
    except Exception as e:
        # Uh oh! Something went wrong while reading our storybook
        print(f"Error reading web log: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # First, let's make a special folder to keep all our stories safe
    # Just like having a special drawer for your favorite books!
    os.makedirs("logs", exist_ok=True)

    # Now let's read our big computer storybook!
    web_logs = parse_web_log('./logs/web_train_log.csv')

    # Time to save all our organized stories in a new book
    if not web_logs.empty:
        # Yay! We found stories to save in our new book!
        web_logs.to_csv('logs/processed_web_logs.csv', index=False)
        print(f"Processed {len(web_logs)} log entries. Saved to logs/processed_web_logs.csv")
    else:
        # Oopsie! We couldn't find any stories to save
        print("No data was processed. Please check the input file.")