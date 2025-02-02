import json
from sklearn.model_selection import train_test_split
from llamaapi import LlamaAPI
import requests
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train_full = pd.read_csv('Data/processed_train.csv')
test_full = pd.read_csv('Data/processed_test.csv')

train_full.dropna(inplace=True)
test_full.dropna(inplace=True)

# My API KEY
API_KEY = "LA-722439f7298347ab8eeddd84bae652ebc82098f0ae814b4eba9582cf7e8246d6"
URL = "https://api.llama-api.com"

llama = LlamaAPI(API_KEY)
MODEL_NAME = "deepseek-v3"

# Checkpoint file
CHECKPOINT_FILE = "checkpoint.json"

# Classification function 
def classify_tweet(tweet):
    api_request_json = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a sentiment classifier. Classify the sentiment of the given tweet as [positive, negative, neutral]. Please respond STRICTLY with the classes (respect lower casing) and not a single more line of text"},
            {"role": "user", "content": f"Tweet: {tweet} \n Sentiment: "}
        ],
        "temperature": 0.1
    }

    response = llama.run(api_request_json)

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    else:
        return f"Error: {response.status_code}, {response.text}"

# For loading the checkpoint
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {}

# For saving the checkpoint
def save_checkpoint(processed_tweets):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(processed_tweets, f, indent=4)

# Selecting a random subset
llama_dataset = train_full.sample(n=100, random_state=42)

# Loading checkpoint
checkpoint = load_checkpoint()
processed_tweets = checkpoint.get("processed_tweets", {})

# List to store results
true_labels = llama_dataset["Sentiment"].tolist()
predicted_labels = []

# Looping through tweets ! 
new_predictions = 0
for index, row in llama_dataset.iterrows():
    tweet = row["Text"]

    # Verify if it has already been processed
    if tweet in processed_tweets:
        print(f"✅ Already processed : {tweet} -> {processed_tweets[tweet]}")
        predicted_labels.append(processed_tweets[tweet])
        continue  # Jumps next, no need to redo work

    try:
        sentiment = classify_tweet(tweet)
        predicted_labels.append(sentiment)
        processed_tweets[tweet] = sentiment

        print(f"📌 Tweet: {tweet} \n🟢 Predict: {sentiment} \n🔵 Real: {row['Sentiment']}\n")

        new_predictions += 1

        # Saves checkpoint every 5 iterations
        if new_predictions % 5 == 0:
            save_checkpoint({"processed_tweets": processed_tweets})
            print("💾 Checkpoint saved !")

        time.sleep(random.uniform(2,6)) # Waits to avoid hitting the Rate Limit
        

    except Exception as e:
        print(f"Error detected ! : {e}. Waiting 10 seconds before retry ...")
        time.sleep(10)

# Save final checkpoint
save_checkpoint({"processed_tweets": processed_tweets})
print("✅ Process finished ! Checkpoint saved .")

# Adds predictions to dataframe
llama_dataset["Predicted"] = predicted_labels

# Calculates performance metrics
accuracy = accuracy_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=["positive", "negative", "neutral"])
report = classification_report(true_labels, predicted_labels, target_names=["positive", "negative", "neutral"])

print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(report)

# Plots confusion matrices
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["positive", "negative", "neutral"],
            yticklabels=["positive", "negative", "neutral"])
plt.xlabel("Predicted")
plt.ylabel("Real")
plt.title(f"Confusion Matrix - Sentiment Analysis ({MODEL_NAME})")
plt.show()
