import os, pickle
from flask import Flask, render_template, request
from transformers import TFAutoModel, AutoTokenizer, TFAutoModelForQuestionAnswering, pipeline
import numpy as np

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Define the identifier for the BERT model
bert_id = 'sebastian-hofstaetter/distilbert-dot-margin_mse-T2-msmarco'

# Load the tokenizer and the model for BERT
bert_tokenizer = AutoTokenizer.from_pretrained(bert_id) 
bert_model = TFAutoModel.from_pretrained(bert_id, from_pt=True)

# Define the identifier for the question answering model
qa_id = 'deepset/minilm-uncased-squad2'

# Load the tokenizer and the model for question answering
qa_tokenizer = AutoTokenizer.from_pretrained(qa_id)
qa_model = TFAutoModelForQuestionAnswering.from_pretrained(qa_id, from_pt=True)

# Create a pipeline for question answering
qa_pipe = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)

# Load the preprocessed contexts
contexts = pickle.load(open('contexts.pkl', 'rb'))

# Function to vectorize the text using BERT
def vectorize_text(text):
    tokenized_text = bert_tokenizer(text[:512], return_tensors='tf')
    vectorized_text = bert_model(tokenized_text)[0][:, 0, :][0]
    return vectorized_text

# Function to calculate the similarity between two vectors
def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))

# Function to order the contexts by similarity to the query
def order_contexts_by_query_similarity(query, contexts):
    # Vectorize the query text
    query_embedding = vectorize_text(query).numpy()

    # Calculate the similarity between the query and each context
    similarities = sorted([
        (vector_similarity(query_embedding, embedding), text) for text, embedding in contexts
    ], reverse=True)

    return similarities

# Function to answer the query using question answering model and ordered contexts
def answer_query(query, max_items=3):
    # Initialize the best score and context
    best_score = 0.0
    best_context = None
    best_start = 0
    best_end = 0

    # Order the contexts by similarity to the query
    best_contexts = order_contexts_by_query_similarity(query, contexts)[:max_items]    

    # Iterate through the best contexts
    for similarity, text in best_contexts:
        # Use the question answering model to find the answer in the context
        result = qa_pipe(question=query, context=text, handle_impossible_answer=True)

        # If a valid answer is found
        if result['start'] != result['end']:
            score = result['score']

            # Update the best score and context if the score is higher
            if score > best_score:
                best_score = score
                best_context = text
                best_start = result['start']
                best_end = result['end']

    # If a valid answer is found, format the answer with highlighted text and confidence score
    if best_score > 0.0:
        return f'{best_context[:best_start]}<mark>{best_context[best_start:best_end]}' \
               f'</mark>{best_context[best_end:]} ({best_score:.1%)}'
    else:
        return 'I don\'t know'

# Define the route for the index page
@app.route('/', methods=['GET'])
def index():
    output = ''
    query = request.args.get('query')

    # Check if a query parameter is present
    if query is None:
        query = ''
    else:
        output = answer_query(query)

    return render_template('index.html', query=query, answer=output)