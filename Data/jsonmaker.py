import pandas as pd 
import json
df = pd.read_csv("Data/labels.csv")
df["prompt"] = "tell me about this image "
df["label"] =" It show : "+ df["text_corrected"].fillna("").astype(str) + "and the overall sentiment here is  " + df["text_corrected"].fillna("").astype(str)
df = df.drop(columns=["text_ocr","x","text_corrected", "overall_sentiment"])
print(df.head(10))
df.to_json('Data/train.jsonl', orient='records', lines=True)
