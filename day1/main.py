from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
model = SentenceTransformer('all-MiniLM-L6-v2')

#Our sentences we like to encode
sentences = ['Hello this is my sentence from llm class 1',
             'This is my sentence from llm class 2',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding[1])
    
sim= cosine_similarity([embeddings[0]], [embeddings[1]])
print(f"Similarity: {sim}")


#Plot the embeddings
plt.hist(embeddings[2])
plt.show()