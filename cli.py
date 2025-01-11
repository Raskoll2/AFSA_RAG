import json
import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI

model = SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

with open('afsa_chunks_embeddings.json', 'r') as f:
    embeddings_data = json.load(f)

chunk_embeddings = []
sections_info = []
chunks_text = []

debug = False

for section_title, section_info in embeddings_data.items():
    chunk_embeddings.extend(section_info['embeddings'])
    sections_info.extend([(section_title, section_info['url'])] * len(section_info['chunks']))
    chunks_text.extend(section_info['chunks'])

chunk_embeddings = torch.tensor(chunk_embeddings).to(device)

def search_query(query):
    while True:
        if query.lower().strip() == 'exit':
            break
        query = "search_query: " + query
        query_embedding = model.encode([query], prompt_name="query", convert_to_tensor=True).to(device)
        similarities = torch.nn.functional.cosine_similarity(query_embedding, chunk_embeddings)
        topk_values, topk_indices = torch.topk(similarities, 5)

        for idx in topk_indices:
            section_title, url = sections_info[idx]
            chunk = chunks_text[idx]
            print(f"Section: {section_title}\nURL: {url}\nRelevant Chunk: {chunk}\n")

        query = input("You: ")


def ai_filtered_query(query, json_data):
    formatted_data = ""
    for i, text in enumerate(json_data):
        formatted_data += f"{i}. {text['section']}\n{text['chunk']}\n\n"

    filter_chat = [
        {"role": "system", "content": f"The following peices of text are potential answers to the following query: {query}." + " Please write a json response with only the relevant indexes of texts. For example, if text n and n+1 was relevant, but all else were off topic, you would responde {\"indexes\": [n, n+1]}."},
        {"role": "user", "content": formatted_data}
    ]

    completion = groq.chat.completions.create(
        model="llama-3.3-8b-8192",
        messages=filter_chat,
        temperature=0.15,
        max_tokens=160,
        stream=False,
        response_format={"type": "json_object"},
    )

    output = completion.choices[0].message.content
    if debug: print(output)
    output = json.loads(output)["indexes"]

    return output

def ask_ai(query):
    main_chat = [
        {"role": "system", "content": "You are a RAG assisted AI model. Your area of expertise is Australian Financial Security Authority (AFSA) procedures and guidelines. Your user is an AFSA employee, you are here to help them find relevant documentation/procedures from your RAG source, and explain it in an intuitive, yet in-depth way. Your RAG ability is triggered when you write \"<RAG>query</RAG>\". You should use the RAG ability when the user wants information that isn't present in your conversation already. When using the RAG ability, it should be the only thing you write in your response, do not address the user and use the ability in the same response!!! This is very important! Do not write anything but the <RAG> query!! You can address the user AFTER the ability has returned relevant information. Give the user links to your sources. The query should be a google searh style query to get the answer to the user's intedned question, not just explicit. Include keywords that you would expect to find in the result."},
    ]

    while True:
        if query.lower().strip() == 'exit':
            break

        main_chat.append({"role": "user", "content": query})

        completion = mistral.chat.completions.create(
            model="mistral-large-latest",
            messages=main_chat,
            temperature=0.3,
            max_tokens=1023,
            top_p=1,
            stream=True,
        )

        response = ""
        ragify = False
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""
            if "<RAG>" in response:
                ragify = True
            elif response == "<R":
                ragify = True
            else:
                print(chunk.choices[0].delta.content or "", end="")


        main_chat.append({"role": "assistant", "content": response})



        if ragify:
            ai_query = response.split("<RAG>")[1].split("</RAG>")[0]

            if debug: print(ai_query)

            query = "search_query: " + ai_query
            query_embedding = model.encode([query], prompt_name="query", convert_to_tensor=True).to(device)
            similarities = torch.nn.functional.cosine_similarity(query_embedding, chunk_embeddings)
            topk_values, topk_indices = torch.topk(similarities, 5)

            data = []
            for i, idx in enumerate(topk_indices):
                section_title, url = sections_info[idx]
                chunk = chunks_text[idx]
                data.append({"section": section_title, "url": url, "chunk": chunk, "index": i})
                if debug: print(f"{i}. {section_title}\n{len(chunk)} chars {chunk[:100]}\n") # working


            response = ai_filtered_query(ai_query, data)

            context = ""
            for i in response:
                context += f"{data[i]['section']}\n{data[i]['url']}\n{data[i]['chunk']}\n\n"

            main_chat.append({"role": "user", "content": context})

            completion = mistral.chat.completions.create(
                model="mistral-large-latest",
                messages=main_chat,
                temperature=0.4,
                max_tokens=2048,
                stream=True,
            )

            response = ""
            print("\n\nAssistant: ")
            for chunk in completion:
                response += chunk.choices[0].delta.content or ""
                print(chunk.choices[0].delta.content or "", end="")

            main_chat.append({"role": "assistant", "content": response})

        print()
        query = input("You: ")






if __name__ == "__main__":
    groq = OpenAI(
        api_key="",
        base_url="https://api.groq.com/openai/v1/"
    )
    mistral = OpenAI(
        api_key="",
        base_url="https://api.mistral.ai/v1"
    )

    ai = True
    user_query = input("\n\nEnter your query (or 'exit' to quit): ")

    if ai:
        ask_ai(user_query)

    else:
        search_query(user_query)
