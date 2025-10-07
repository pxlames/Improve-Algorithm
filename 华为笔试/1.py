import math
from collections import defaultdict

def tokenize(text):
    """
    Splits a string into a list of words, converting to lowercase.
    """
    return text.lower().split()

def compute_tf(document):
    """
    Computes Term Frequency (TF) for a document.
    Returns a dictionary of word -> frequency.
    """
    tf = defaultdict(int)
    words = tokenize(document)
    for word in words:
        tf[word] += 1
    return tf

def compute_idf(corpus):
    """
    Computes Inverse Document Frequency (IDF) for the entire corpus.
    Returns a dictionary of word -> idf_score.
    """
    idf = defaultdict(int)
    N = len(corpus)
    
    # First, count document frequency for each word
    doc_freq = defaultdict(int)
    for doc in corpus:
        unique_words = set(tokenize(doc))
        for word in unique_words:
            doc_freq[word] += 1
            
    # Then, calculate IDF score
    for word, count in doc_freq.items():
        idf[word] = math.log((N + 1) / (count + 1)) + 1
        
    return idf

def get_historical_window(corpus, query_time, window_size):
    """
    Returns the documents within the historical window.
    The window includes documents from timestamp 0 up to query_time,
    with a max size of `window_size`.
    """
    window_end_index = query_time
    
    # Determine the start of the window
    window_start_index = max(0, window_end_index - window_size + 1)

    # Slice the corpus to get the historical window
    historical_window = corpus[window_start_index : window_end_index + 1]
    
    return historical_window, window_start_index

def solve():
    """
    Main function to handle input, process queries, and print output.
    """
    try:
        # Read corpus size
        N_str = input()
        if not N_str: return
        N = int(N_str)
        
        # Read the corpus documents
        corpus = [input() for _ in range(N)]
        
        # Read window size
        K = int(input())
        
        # Read number of queries
        P = int(input())
        
        # Pre-compute IDF for the entire corpus
        global_idf = compute_idf(corpus)
        
        results = []
        for _ in range(P):
            query_line = input()
            query_time, query_content = query_line.split(" ", 1)
            query_time = int(query_time)
            
            # Get the historical window and its start index
            historical_window, window_start_index = get_historical_window(corpus, query_time, K)
            
            if not historical_window:
                results.append("-1")
                continue
                
            # Compute query vector (A)
            query_tf = compute_tf(query_content)
            query_vector = defaultdict(float)
            for word, freq in query_tf.items():
                query_vector[word] = freq * global_idf.get(word, 0)
            
            max_similarity = -1.0
            best_doc_index = -1
            
            # Iterate through the documents in the historical window
            for i, doc_content in enumerate(historical_window):
                current_doc_index = window_start_index + i
                
                # Dynamic weight based on position in the window
                dynamic_weight = (i + 1) / K
                
                # Compute document vector (B)
                doc_tf = compute_tf(doc_content)
                doc_vector = defaultdict(float)
                for word, freq in doc_tf.items():
                    # IDF is for the full corpus, but TF is for the current doc.
                    doc_vector[word] = freq * global_idf.get(word, 0) * dynamic_weight
                
                # Cosine Similarity (A · B) / (||A|| * ||B||)
                dot_product = 0.0
                norm_A = 0.0
                norm_B = 0.0
                
                # Calculate dot product and norms
                all_words = set(query_vector.keys()) | set(doc_vector.keys())
                for word in all_words:
                    dot_product += query_vector.get(word, 0) * doc_vector.get(word, 0)
                    
                for word in query_vector:
                    norm_A += query_vector[word] ** 2
                
                for word in doc_vector:
                    norm_B += doc_vector[word] ** 2
                
                norm_A = math.sqrt(norm_A)
                norm_B = math.sqrt(norm_B)
                
                similarity = 0.0
                if norm_A > 1e-6 and norm_B > 1e-6:
                    similarity = dot_product / (norm_A * norm_B)
                
                # Check for tie and similarity threshold
                if similarity >= 0.6:
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_doc_index = current_doc_index
                    elif similarity == max_similarity:
                        # Tie-breaking: choose the earlier document
                        if best_doc_index == -1 or current_doc_index < best_doc_index:
                            best_doc_index = current_doc_index
            
            results.append(str(best_doc_index))
            
        print(" ".join(results))
    
    except (IOError, ValueError) as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    solve()