# AFSA RAG

A RAG system for the Australian Financial Security Authority. It scrapes their public predecures and guides wesbite, but can be expanded to internal precedures. An efficient embeddings model is used to store chunks of the data in a vector database. The `cli.py` file uses a large model to generate a query, retrieves a few chunks, uses a small model to filter them, and then the large model respondes. It needs a Mistral and a Groq API key but you can change the base_url and use whatever models + providers you want. 

The API doesn't include small model filtering, just the basic RAG functionality. It uses Flask and is not particularly production ready.

`pip install torch sentence_transformers openai bs4 requests flask`
