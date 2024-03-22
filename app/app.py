from flask import Flask, request, jsonify, send_from_directory, url_for, Response, stream_with_context
from flask_cors import CORS # comment this on deployment
from flask_restful import reqparse
import pandas as pd
import io
from umap import UMAP
from metric_learn import NCA
from sklearn.manifold import TSNE


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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



class ClassifierModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ClassifierModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    def get_activation(self, x):
        # Function to extract activations from the first layer
        with torch.no_grad():
            activation = self.relu(self.fc1(x))
        return activation




from openai import OpenAI 
client = OpenAI(api_key = '')

try:
     _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
     pass
else:
    ssl._create_default_https_context = _create_unverified_https_context



app = Flask(__name__, static_url_path='', static_folder='build')
cors = CORS(app)
# logging.basicConfig(filename='flask.log', level=logging.DEBUG)
should_continue = {"flag": True}

# Serve home route
@app.route("/") 
def home():
    return send_from_directory(app.static_folder, "index.html")
# Performs selected dimensionality reduction method (reductionMethod) on uploaded data (data), considering selected parameters (perplexity, selectedCol)
# Simulated in-memory storage for demonstration purposes
in_memory_storage = {}

@app.route("/initialize-embeddings", methods=["POST"])
def initialize_batch():
    content = request.json
    dataset = content.get('dataset')
    theme = content.get('theme')

    should_continue["flag"] = True  # Make sure to reset the flag when starting

    # Simulate storing the initialization data with a unique session ID
    session_id = "session_123"  # In a real application, generate a unique ID
    in_memory_storage[session_id] = {"dataset": dataset, "theme": theme}

    # Return session ID to the client
    return jsonify({"message": "Processing initialized", "status": "success", "session_id": session_id})

@app.route("/modify-embeddings/", methods=["GET"])
def modify_batch():
    # Retrieve session ID from query parameter
    session_id = request.args.get('session_id')
    if not session_id or session_id not in in_memory_storage:
        return jsonify({"message": "Session ID is invalid or missing", "status": "error"}), 400

    # Retrieve stored data using session ID
    session_data = in_memory_storage.get(session_id)
    dataset = session_data.get('dataset')
    theme = session_data.get('theme')
    n = 5#int(session_data.get('batchsize'))

    attribute = theme

    # df = pd.read_csv('src/datasets/f_emb.csv') # paper's main topic, methodology, type of evauluation, data used or intended audience
    df = pd.read_csv('src/datasets/'+dataset+'_emb.csv')
#If the attribute is not present in a sentence, assign -1.
    prompt_template = """
    You are tasked with analyzing a list of texts to classify each one according to a specific attribute provided. Your goal is to assign an integer label to each paper based on this attribute. Each unique attribute value found in the texts should correspond to a unique integer, starting from 0. It's important that the attribute values you extract are simple, concise and easily understandable.

    For this task, you will also be given a previously established mapping of integers to attribute values if available. If you encounter a new attribute value not present in the existing mapping, you should extend the mapping by assigning a new integer to this value. However, do not alter the original mapping.
 
    Your output should include both an array of classification labels for the texts and the (potentially updated) mapping of integers to attribute values. Ensure that every text is classified, and the size of the output labels array matches the number of input texts. The output should be formatted as JSON.

    Here are the specific details for this task:
    List of texts: {{sentences}}
    Attribute to classify by: {{attribute}}
    Previously used mapping: {{mapping}}

    Ensure that each text is accounted for in the output. The output should be JSON in the following format:

    {
    'labels': [integer1, integer2, ..., integerN],  # N equals the number of papers
    'mapping': {integer1: 'label1', integer2: 'label2', ...}
    }
 
    where each integer in 'labels' represents the classification of the corresponding text based on the attribute "{attribute}", and 'mapping' shows the association between integers and attribute values. It is essential that the 'labels' array includes a label for every paper provided, ensuring no text is missed.

    Please think carefully and ensure the response adheres to these instructions.

    Output:
    """  

    sentences = df['text'].tolist()

    def useNCA(targets_all):
        targets_all = np.array(targets_all).flatten()

        nca = NCA(random_state=42,max_iter=500, n_components=2)
        nca.fit(df.iloc[:len(targets_all), 0:768], targets_all+1)
        embedding = nca.transform(df.iloc[:,0:768])
        print(embedding.shape)

        target_labels = np.concatenate([targets_all, np.full((len(df) - len(targets_all) ), -2)]).flatten()
        
        return embedding,target_labels
    

    def makeProjections(embedding,target_labels):
        #reducer= UMAP(n_components=2)
        #reducer = TSNE(random_state=42,n_components=2,perplexity=100)
        #embedding = reducer.fit_transform(embedding)

        # clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
        #clusterer = KMeans(n_clusters=3, random_state=42)
        clusterer.fit(embedding)
        labels = clusterer.labels_


        df_mod = pd.DataFrame(embedding)
        df_mod['text'] = df['text']
        df_mod['cluster']= target_labels
        #df_mod['cluster']= labels

        #df_emb = df_emb.join(df_features)
        df_mod = df_mod.join(df.iloc[:,768:-1])

        return df_mod
    

    def useNN(targets_all):
        targets_all = np.array(targets_all).flatten()
        input_size = 768
        hidden_size = 50
        num_classes = len(set(targets_all))+1
        num_examples = len(targets_all)
        embeddings_tensor = torch.tensor(df.iloc[:len(targets_all), 0:768].values).float()
        labels = torch.tensor(targets_all+1)

        batch_size = 32  # You can adjust the batch size
        dataset = TensorDataset(embeddings_tensor, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model, loss, and optimizer
        model = ClassifierModel(input_size,hidden_size, num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)


        # Training loop
        num_epochs = 200  # You can adjust the number of epochs
        for epoch in range(num_epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                #print(targets)
                loss = criterion(outputs, targets)
                loss.backward()              
                optimizer.step()
            #if (epoch+1)%50==0:
            #    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
            
        embeddings_all = torch.tensor(df.iloc[:, 0:768].values).float()
        embedding = model.get_activation(embeddings_all).detach().numpy()
        target_labels = np.concatenate([targets_all, np.full((len(df) - len(targets_all) ), -2)]).flatten()

        return embedding,target_labels

    def generate_updates():
        n_total = len(df)
        targets_all = []
        mapping_all = []
        mapping={}  
        i = 0 
        while i < int(n_total/n):
            if not should_continue["flag"]:
                break  # Stop the loop if the flag is False
            texts = sentences[n*i:n*(i+1)]
            formatted_sentences = '\n'.join([f"- {sentence}" for sentence in texts])
            prompt = prompt_template.replace("{{sentences}}", formatted_sentences).replace("{{attribute}}", attribute).replace("{{mapping}}", str(mapping))

            result = getScoreJSON(prompt, attribute) 
            
            try:
                data = json.loads(result)
                targets = data['labels']
                print(targets)

                
            except: 
                targets = []
                
            try:
                print(data['mapping'])
                mapping.update(data['mapping'])
            except:
                yolo = 0            
            print(n*i,n*(i+1),len(targets))
 

            if len(targets)>=len(texts):
                targets_all.append(targets[:n])
                mapping_all.append(mapping)
                embedding, target_labels = useNCA(targets_all) 
                df_mod = makeProjections(embedding,target_labels+2)
                df_mod['GPTLabelName'] = [mapping[str(i)] if (i >=0) else "Unavailable" for i in target_labels]
                df_clstr = get_cluster_labels(df_mod, theme)
 
                if i >= 3:#int(n_total/n/2):
                    yield f"data: {json.dumps({'update': (i+1)/int(n_total/n), 'embeddings':json.loads(df_mod.to_json(orient='values')), 'labels': json.loads(df_clstr.to_json(orient='values')),'mapping':mapping})}\n\n"
                else:
                    yield f"data: {json.dumps({'update': (i+1)/int(n_total/n), 'embeddings':'none', 'mapping': mapping})}\n\n"

                i=i+1
        # Indicate completion
        yield f"data: {json.dumps({'status': 'Completed', 'embeddings':json.loads(df_mod.to_json(orient='values')), 'labels': json.loads(df_clstr.to_json(orient='values')), 'mapping': mapping,'message': 'All batches processed'})}\n\n"

    return Response(stream_with_context(generate_updates()), content_type='text/event-stream')

@app.route("/stop-processing", methods=["POST"])
def stop_processing():
    should_continue["flag"] = False  # Set the flag to False to stop the loop
    return jsonify({"message": "Processing will be stopped"})

def getGPTresponse(prompt, theme):
    
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": "You are an expert in comparing and analyzing "+ theme+"."},
            {"role": "user", "content": prompt},
        ],
        temperature= 0.4
    )
    
    try:
        result = response.choices[0].message.content
    except:
        result = []
    
    return result

def getScoreJSON(prompt, theme):
    
    response = client.chat.completions.create(
        model= "gpt-4-0125-preview",
        response_format={ "type": "json_object" },

        messages=[
            {"role": "system", "content": "You are an expert in comparing and analyzing "+ theme+"."},
            {"role": "user", "content": prompt},
        ],
        temperature= 0.4
    )
    
    try:
        result = response.choices[0].message.content
    except:
        result = []
    
    return result

def get_cluster_labels(df, theme):

    # df = df.sample(frac = 1)
    df = df.copy()  # similar to above but without shuffling
    def aggregate_texts(group):
        return ', '.join(group['text'].tolist())

    clustered_texts = df.groupby('cluster').apply(aggregate_texts).reset_index(name='aggregated_text')

    cluster_labels = []

    for i in range(1,len(clustered_texts)):
    
        cluster_id, text_data = i, clustered_texts.iloc[i]['aggregated_text'][:8000]
        prompt = f"Given the provided text data:{text_data}, please generate a concise and informative label that captures the essence of the content in relation to a specified theme. Focus on specific aspects rather  than broad or generic terms and topics, but keep the label simple and understandable and such that multiple texts could be attributed to it. Ensure the label directly reflects the central or unique attribute discussed in the texts. The theme is {theme}"

        parameters = {'model': 'gpt-4-0125-preview', 'messages': 
                      [{"role": "system", "content": "You are an expert text label provider that provides labels for the texts provided. The labels should relate on to the attribute:" +theme}, 
                       {"role": "user", "content": prompt},]}
        #response = client.chat.completions.create(**parameters)
        lbl = "none"#response.choices[0].message.content
        #print(f"Cluster {cluster_id} Summary: " + lbl)

        cluster_labels.append(lbl)
        #time.sleep(1)

        # cluster_labels.append(f"label {i}")  # 🧪 testing only

    df_clstr = df.groupby('cluster')[[0,1]].mean().iloc[1:]
    df_clstr['label'] = cluster_labels

    return df_clstr


if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=8000)