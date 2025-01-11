from flask import Flask, request, jsonify
import json
import torch
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

model = SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

with open('afsa_chunks_embeddings.json', 'r') as f:
    embeddings_data = json.load(f)

chunk_embeddings = []
sections_info = []
chunks_text = []

# Flatten the embeddings and store the chunks
for section_title, section_info in embeddings_data.items():
    chunk_embeddings.extend(section_info['embeddings'])
    sections_info.extend([(section_title, section_info['url'])] * len(section_info['chunks']))
    chunks_text.extend(section_info['chunks'])

# Convert embeddings back to tensor and move to the appropriate device
chunk_embeddings = torch.tensor(chunk_embeddings).to(device)


def search_query(query, num_results):
    query = "search_query: " + query
    query_embedding = model.encode([query], prompt_name="query", convert_to_tensor=True).to(device)
    similarities = torch.nn.functional.cosine_similarity(query_embedding, chunk_embeddings)
    topk_values, topk_indices = torch.topk(similarities, num_results)
    results = []
    for idx, similarity in zip(topk_indices, topk_values):
        section_title, url = sections_info[idx]
        chunk = chunks_text[idx]
        if len(chunk) > 500:
            results.append({
                "title": section_title,
                "url": url,
                "similarity": float(similarity),
                "content": chunk
            })
    return results

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get("query", "")
    num_results = data.get("num_results", 3)

    results = search_query(query, num_results)

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
