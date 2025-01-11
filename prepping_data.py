import requests
from bs4 import BeautifulSoup
import json


# Scrape links from this index page
# Links point to pages with the actual data
url = "https://www.afsa.gov.au/professionals/resource-hub/practice-guidance"

response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', class_='views-table')

    data = []

    for row in table.find_all('tr'):
        link_cell = row.find('td', class_='views-field views-field-title')

        if link_cell:
            link_tag = link_cell.find('a')
            if link_tag:
                link = "https://www.afsa.gov.au" + link_tag['href']
                title = link_tag.text.strip()
                data.append({
                    'title': title,
                    'link': link
                })

    # Save the data to a JSON file
    with open('afsa_links.json', 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Successfully saved {len(data)} links to afsa_links.json")

else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")



# -------------------------------- Scrape the actual data --------------------------------
with open('afsa_links.json', 'r') as f:
    links_data = json.load(f)

def scrape_subsections(url):
    try:
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the ordered list with class "ordered-custom"
            ol = soup.find('ol', class_='ordered-custom')
            if ol:
                chunks = []

                for li in ol.find_all('li'):
                    # Convert each <li> content into plain text (markdown)
                    chunk = li.get_text(strip=True)
                    chunks.append(chunk)

                return chunks
        else:
            print(f"Failed to retrieve the page at {url}. Status code: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


# Chunk the data
final_data = {}

total = len(links_data)
i = 0
output = ""
for entry in links_data:
    i += 1
    print(str(round(i*100/total)) + "%")
    section_title = entry['title']
    section_url = entry['link']

    subsections = scrape_subsections(section_url)

    chunkierChunks = []

    try:
        output = ""
        for chunk in subsections:
            output += chunk + "\n\n"
            if len(output) > 4000:
                chunkierChunks.append(output)
                output = ""

        # Check if remainder chunk is less than 400 characters
        if len(output) < 400:
            if chunkierChunks:
                chunkierChunks[-1] += output
            else:
                chunkierChunks.append(output)
        else:
            chunkierChunks.append(output)

    except:
        print(subsections)

    if subsections:
        final_data[section_title] = {
            'url': section_url,
            'chunks': chunkierChunks,
        }

with open('afsa_sections_with_chunks.json', 'w') as f:
    json.dump(final_data, f, indent=4)

print(f"Successfully saved the sections data with chunks to afsa_sections_with_chunks.json")



# -------------------------------- Embed the data --------------------------------
import json
from sentence_transformers import SentenceTransformer

# This model is pretty good + efficient
model = SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True)

with open('afsa_sections_with_chunks.json', 'r') as f:
    data = json.load(f)

embeddings_data = {}

# Iterate over the sections and embed the chunks
total = len(data)
i = 0
for section_title, section_info in data.items():
    i += 1
    print(round(i*100/total))
    chunks_ = section_info['chunks']

    chunks = []

    for chunk in chunks_:
        if len(chunk) > 500:
            chunks.append(chunk)

    # prompt_name="document" tells the model there's it's not the prompt
    chunk_embeddings = model.encode(chunks, prompt_name="document", convert_to_tensor=True)

    embeddings_data[section_title] = {
        'url': section_info['url'],
        'chunks': chunks,
        'embeddings': chunk_embeddings.tolist()
    }

with open('afsa_chunks_embeddings.json', 'w') as f:
    json.dump(embeddings_data, f, indent=4)

print("Embeddings successfully saved to afsa_chunks_embeddings.json")
