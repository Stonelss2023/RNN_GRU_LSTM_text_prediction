from DL6_3_1jzlyrics_downloader import download_data
from DL6_3_2jzlyrics_processor import process_data
from DL6_3_3jzlyrics_sampler import data_iter_random, data_iter_consecutive

class JaychouDataset:
    def __init__(self, num_chars=10000):
        self.num_chars = num_chars
        self.corpus_chars = None
        self.idx_to_char = None
        self.char_to_idx = None
        self.vocab_size = None
        self.corpus_indices = None

    def load_data(self):
        if self.corpus_chars is None:
            print("Loading data...")
            self.corpus_chars = download_data()
            (self.corpus_chars, 
             self.idx_to_char, 
             self.char_to_idx, 
             self.vocab_size, 
             self.corpus_indices
             ) = process_data(self.corpus_chars, self.num_chars)
            print("Data loaded and processed.")

    def get_random_iter(self, batch_size, num_steps):
        self.load_data()
        return data_iter_random(self.corpus_indices, batch_size, num_steps)

    def get_consecutive_iter(self, batch_size, num_steps):
        self.load_data()
        return data_iter_consecutive(self.corpus_indices, batch_size, num_steps)

    def get_corpus_chars(self):
        self.load_data()
        return self.corpus_chars

    def get_vocab_info(self):
        self.load_data()
        return self.idx_to_char, self.char_to_idx, self.vocab_size

    def get_corpus_indices(self):
        self.load_data()
        return self.corpus_indices