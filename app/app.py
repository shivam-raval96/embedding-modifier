from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS # comment this on deployment
from flask_restful import reqparse
import pandas as pd
import io
from umap import UMAP
# import logging

from sklearn.cluster import KMeans
import hdbscan
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim

import json
import random
import re
from rake_nltk import Metric, Rake
import ssl
import os
import time

from openai import OpenAI
client = OpenAI(api_key = '')

try:
     _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
     pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Simple neural network with one hidden layer
class TransformationNet(nn.Module):
        def __init__(self, input_size, output_size):
            super(TransformationNet, self).__init__()
            self.fc1 = nn.Linear(input_size, output_size)
        
        def forward(self, x):
            x = self.fc1(x)
            
            return x

# Hyperparameters
input_size = 768
output_size = 768
learning_rate = 0.01
num_epochs = 5000  # default 10000
weight_decay = 0.001  # L2 regularization strength

app = Flask(__name__, static_url_path='', static_folder='build')
cors = CORS(app)
# logging.basicConfig(filename='flask.log', level=logging.DEBUG)

# Serve home route
@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")
# Performs selected dimensionality reduction method (reductionMethod) on uploaded data (data), considering selected parameters (perplexity, selectedCol)
@app.route("/modify-embeddings", methods=["POST"])
def modify():

    #data = request.json['data']
    dataset = request.json['dataset']
    theme = request.json['theme']
    print("Cluster by:", theme)

    # df = pd.read_csv('src/datasets/f_emb.csv')
    df = pd.read_csv('src/datasets/'+dataset+'_emb.csv')

    prompt ="Construct a scale for "+ theme+ " going from 0 to 10. I am providing you with a two sentences. I want you to provide a number \
    that quantifies the difference between the scores for "+ theme+ " detected in the text using the constructed scale. Only provide a score and nothing else. \
    Here are the two sentences: "
    n = 8
    ids = np.random.randint(0,len(df),n)
    scores = np.zeros((n,n))

    for i in range(n):
        for j in range(i):
        
            result = get_score(df['text'][ids[i]], df['text'][ids[j]], prompt, theme)
        
            scores[i,j] = result

    scores = np.tril(scores).T+np.tril(scores)
    #np.fill_diagonal(scores, 10)

    transformed = learn_transformation(scores,df,ids)

    # dimensionality reduction
    umap = UMAP(n_components=2, n_neighbors=10,min_dist=0)
    embedding = umap.fit_transform(transformed,)
    
    # clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=7, gen_min_span_tree=True)
    #clusterer = KMeans(n_clusters=3, random_state=42)
    clusterer.fit(embedding)
    labels = clusterer.labels_
   
    
    df_mod = pd.DataFrame(embedding)
    df_mod['text'] = df['text']
    df_mod['cluster'] = labels
    df_mod = df_mod.join(df.iloc[:,-6:])

    # label clusters
    df_clstr = get_cluster_labels(df_mod, theme)

    return {"embeddings": json.loads(df_mod.to_json(orient='values')),
            "labels": json.loads(df_clstr.to_json(orient='values'))}

    # return jsonify(data, theme)     # 🧪 testing only

def get_score(text1, text2, prompt, theme):
    
    response = client.chat.completions.create(
        model="gpt-4",
        # model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in comparing sentences based on "+ theme+"."},
            {"role": "user", "content": prompt +text1 +', '+text2},
        ]
    )
    print(response.choices[0].message.content)
    
    try:
        result = json.loads(response.choices[0].message.content)
    except:
        result = []
    
    return result

    # testing_response = np.random.randint(1,10)  # 🧪 testing only
    # print(testing_response)                     # 🧪 testing only
    # return np.random.randint(1,10)              # 🧪 testing only

def distance(outputs):
    n = len(outputs)
    pred_score = torch.zeros((n,n))

    for i in range(n):
        for j in range(n):
            
            pred_score[i,j] = torch.norm(outputs[i]-outputs[j])
        
    return pred_score

def learn_transformation(scores, df, ids):
    
    print('hello', num_epochs)
    # Initialize network and optimizer with weight_decay for L2 regularization
    net = TransformationNet(input_size, output_size)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    target_matrix = torch.tensor(scores).float()

    # Training
    for epoch in range(num_epochs):
        # Flatten the source matrix and pass it through the network
        outputs = net(torch.tensor(df.iloc[ids,0:768].to_numpy()).float())
        loss = criterion(distance(outputs), target_matrix)#criterion(outputs@outputs.T, target_matrix)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        transformed = net(torch.tensor(df.iloc[:,0:768].to_numpy()).float()).detach().numpy()
    
    return transformed

def get_cluster_labels(df, theme):

    # df = df.sample(frac = 1)
    df = df.copy()  # similar to above but without shuffling

    def aggregate_texts(group):
        return ', '.join(group['text'].tolist())

    clustered_texts = df.groupby('cluster').apply(aggregate_texts).reset_index(name='aggregated_text')

    cluster_labels = []

    for i in range(1,len(clustered_texts)):
    
        cluster_id, text_data = i, clustered_texts.iloc[i]['aggregated_text'][:8000]
        prompt = f"Here is a corpus of text: {text_data}. Can you provide a label for the text based on the common theme that is short and informative? Make sure that the label contains a only one short phrase of maximum two words and effectively summarizes the texts."

        parameters = {'model': 'gpt-4', 'messages': 
                      [{"role": "system", "content": "You are an expert text summarizer."}, 
                       {"role": "user", "content": prompt},]}
        response = client.chat.completions.create(**parameters)
        lbl = response.choices[0].message.content
        print(f"Cluster {cluster_id} Summary: " + lbl)

        cluster_labels.append(lbl[1:-1])
        time.sleep(1)

        # cluster_labels.append(f"label {i}")  # 🧪 testing only

    df_clstr = df.groupby('cluster')[[0,1]].mean().iloc[1:]
    df_clstr['label'] = cluster_labels

    return df_clstr


if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=8000)