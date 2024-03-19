from flask import Flask, request, jsonify, send_from_directory, url_for
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

from openai import OpenAI
client = OpenAI(api_key = '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')

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
def modify_batch():

    #data = request.json['data']
    dataset = request.json['dataset']
    theme = request.json['theme']
    print("Cluster by:", theme)
    attribute = theme

    # df = pd.read_csv('src/datasets/f_emb.csv')
    df = pd.read_csv('src/datasets/'+dataset+'_emb.csv')

    prompt_template = """
    Given a list of sentences and an attribute to classify by, assign an integer label to each sentence based on the specified attribute. The attribute could be anything, such as color, animal, action, etc., and each unique value of the attribute in the sentences should be assigned a distinct integer starting from 0. If the attribute is not present in a sentence, assign -1.


    List of sentences: {{sentences}}
    Attribute to classify by: {{attribute}}
    Output: {'labels':[integer1, integer2, ...], 'mapping':{integer1: 'label1', integer2:' label2', ... }} where each integer represents the classification of the corresponding sentence based on the attribute "{attribute}".


    Classify each sentence based on the attribute and output the classification labels and the mapping as a JSON output. Only respond with the JSON array of classification labels and the mapping. Note: The integer assignment to each classification should be consistent across the output array. Make sure that the output array is of the same size as the number of sentences. Think carefully and respond.


    Output:
    """


    prompt_template3 = """
    Given a list of sentences and an attribute to classify by, your task is to assign an integer label to each sentence based on the specified attribute. The attribute could be anything, such as color, animal, action, etc. For each unique value of the attribute in the sentences, assign a distinct integer starting from 0. If the attribute is not present in a sentence, assign -1. It is crucial that the output array of labels matches the size of the input list of sentences exactly.

    You are also provided with a previously used mapping of integers to attribute values. Extend this mapping ONLY if new labels of the attribute are found in the sentences but never change the original mapping provided. The final output should include both the array of classification labels and the updated mapping.

    List of sentences: {{sentences}}
    Attribute to classify by: {{attribute}}
    Previously used mapping: {{mapping}}

    Ensure that each sentence is accounted for in the output. The output should be JSON in the following format:

    {
    'labels': [integer1, integer2, ..., integerN],  # N equals the number of sentences
    'mapping': {integer1: 'label1', integer2: 'label2', ...}
    }

    where each integer in 'labels' represents the classification of the corresponding sentence based on the attribute "{attribute}", and 'mapping' shows the association between integers and attribute values. It is essential that the 'labels' array includes a label for every sentence provided, ensuring no sentence is missed.

    Please think carefully and ensure the response adheres to these instructions.

    Output:
    """
    n_total = len(df)
    n = 20
    targets_all = []
    mapping_all = []
    mapping={}
    i = 0

    sentences = df['text'].tolist()
    while i < int(n_total/n):
        texts = sentences[n*i:n*(i+1)]
        formatted_sentences = '\n'.join([f"- {sentence}" for sentence in texts])
        if i ==0:
            prompt = prompt_template.replace("{{sentences}}", formatted_sentences).replace("{{attribute}}", attribute)
        else:
            prompt = prompt_template3.replace("{{sentences}}", formatted_sentences).replace("{{attribute}}", attribute).replace("{{mapping}}", str(mapping))

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


        if len(targets)==n:
            targets_all.append(targets)
            mapping_all.append(mapping)
            i=i+1


    targets_all = np.array(targets_all).flatten()

    nca = NCA(random_state=42,max_iter=500, n_components=2)
    nca.fit(df.iloc[:int(n_total), 0:768], targets_all+1)
    embedding = nca.transform(df.iloc[:,0:768])

    #reducer= UMAP(n_components=2)
    reducer = TSNE(n_components=2,perplexity=50)
    embedding = reducer.fit_transform(embedding)

    # clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
    #clusterer = KMeans(n_clusters=3, random_state=42)
    clusterer.fit(embedding)
    labels = clusterer.labels_


    df_mod = pd.DataFrame(embedding)
    df_mod['text'] = df['text']
    df_mod['cluster']= labels
    named_targets = []
    mapping['-1'] = 'none'
    print(mapping)
    for i in targets_all:
        #try:
            named_targets.append(mapping[str(i)])
        #except:
            #named_targets.append('none')
    print()
    df_mod['target']= named_targets ###3

    #df_emb = df_emb.join(df_features)
    #df_mod = df_mod.join(df.iloc[:,-6:])

    # label clusters 
    df_clstr = get_cluster_labels(df_mod, theme)

    return {"embeddings": json.loads(df_mod.to_json(orient='values')),
            "labels": json.loads(df_clstr.to_json(orient='values'))}

    # return jsonify(data, theme)     # 🧪 testing only


def modify_classification():

    #data = request.json['data']
    dataset = request.json['dataset']
    theme = request.json['theme']
    print("Cluster by:", theme)

    # df = pd.read_csv('src/datasets/f_emb.csv')
    df = pd.read_csv('src/datasets/'+dataset+'_emb.csv')

    scaleprompt ="Construct a detailed scale for "+ theme+ " going from 0 to 10. Be as fine grained as you need to be. Return only a json dictionary."
    scale = getGPTresponse(scaleprompt, theme)
    prompt ="Here is a scale for "+theme+ ": "+scale+" I am providing you with a sentence. I want you to provide a number \
    that quantifies the "+ theme+ " detected in the text using the provided scale, interpolate if needed, but the score has to be an integer. You MUST only provide a score and nothing else. If you can't detected anything, return 100.\
    Here is the sentence: "
    '''prompt ="Construct a scale for "+ theme+ " going from 0 to 10.  I am providing you with a sentence. I want you to provide a number \
    that quantifies the "+ theme+ " detected in the text using the provided scale, interpolate if needed, but the score has to be an integer. You MUST only provide a score and nothing else. If you can't detected anything, return 100.\
    Here is the sentence:"'''
    n = 50
    #ids = np.random.choice(np.linspace(0,len(df)-1,len(df)), size=n, replace=False).astype(int)
    targets = np.zeros(n)

    for i in range(n):
            result = getScore(df['text'][i], prompt, theme) #getScorewithScale(df['text'][i], promptWscale, theme) 
            targets[i] = result



    nca = NCA(random_state=42,max_iter=300, n_components=2)
    nca.fit(df.iloc[:n, 0:768], targets)
    embedding = nca.transform(df.iloc[:,0:768])

    #umap = UMAP(n_components=2)
    #embedding = umap.fit_transform(embedding)
    
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

def modify_pairwise():

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

def getGPTresponse(prompt, theme):
    
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": "You are an expert in comparing and analyzing "+ theme+"."},
            {"role": "user", "content": prompt},
        ],
        temperature= 0.0
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
        temperature= 0.0
    )
    
    try:
        result = response.choices[0].message.content
    except:
        result = []
    
    return result
def getScore(text,prompt, theme):
    
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": "You are an expert in comparing and analyzing "+ theme+"."},
            {"role": "user", "content": prompt+text},
        ],
        temperature= 0.0
    )
    
    try:
        result = response.choices[0].message.content
    except:
        result = []
    
    return result

def get_score(text1, text2, prompt, theme):
    
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
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
        prompt = f"Here is a corpus of text: {text_data}. Can you provide a label for the text based on the {theme} that is short and informative? Make sure that the label contains a only one short phrase of maximum two words and effectively summarizes the texts."

        parameters = {'model': 'gpt-4-0125-preview', 'messages': 
                      [{"role": "system", "content": "You are an expert text label provider that provides labels for the texts provided. The labels should relate on to the attribute:" +theme}, 
                       {"role": "user", "content": prompt},]}
        response = client.chat.completions.create(**parameters)
        lbl = response.choices[0].message.content
        print(f"Cluster {cluster_id} Summary: " + lbl)

        cluster_labels.append(lbl)
        time.sleep(1)

        # cluster_labels.append(f"label {i}")  # 🧪 testing only

    df_clstr = df.groupby('cluster')[[0,1]].mean().iloc[1:]
    df_clstr['label'] = cluster_labels

    return df_clstr


if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=8000)