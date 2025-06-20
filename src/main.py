from data_loader import BioASQDataLoader
from vector_store import VectorStore

def main():
    # Initialize components
    data_loader = BioASQDataLoader()
    vector_store = VectorStore()
    
    # Load data
    print("Loading corpus...")
    corpus = data_loader.load_corpus()
    
    # Setup vector store
    print("Creating collection...")
    vector_store.create_collection()
    
    print("Inserting passages...")
    vector_store.insert_passages(corpus)

if __name__ == "__main__":
    main() 