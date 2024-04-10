import pickle
import gzip

# Path to the original pickle file
original_file_path = 'similarity.pkl'

# Path to the compressed file
compressed_file_path = 'compress.pkl.gz'

# Load the data from the original pickle file
with open(original_file_path, 'rb') as f:
    data = pickle.load(f)

# Compress the data and write it to a gzip-compressed file
with gzip.open(compressed_file_path, 'wb') as f:
    pickle.dump(data, f)

print(f'File compressed and saved as {compressed_file_path}')
