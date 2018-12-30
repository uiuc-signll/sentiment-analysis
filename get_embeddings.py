from get_reviews import get_reviews
from sacremoses import MosesTokenizer, MosesDetokenizer
from allennlp.modules.elmo import batch_to_ids, Elmo
import torch

BATCH_SIZE = 512

def get_tokenized_review_list():
    mt = MosesTokenizer()
    reviews = get_reviews()[0]
    tokenized_list = [mt.tokenize(review_text, escape=False) for review_text in reviews]
    return (tokenized_list, reviews[1])

def charids_to_file():
    tokenized_list = get_tokenized_review_list()
    print(len(tokenized_list))
    print('Tokenized reviews')
    
    character_ids = batch_to_ids(tokenized_list)
    print('Converted text to character ids')
    torch.save(character_ids, 'char_ids_{0}.pt'.format(str(i)))

def get_embeddings():
    tokenized_list = get_tokenized_review_list()[0]
    print(len(tokenized_list))
    print('Tokenized reviews')
    
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    character_ids = batch_to_ids(tokenized_list)
    print('Converted text to character ids')

    elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)
    embeddings = elmo(character_ids)
    
    # embeddings['elmo_representations'] is length two list of tensors.
    # Each element contains one layer of ELMo representations with shape
    # (2, 3, 1024).
    #   2    - the batch size
    #   3    - the sequence length of the batch
    #   1024 - the length of each ELMo vector
    return (embeddings['elmo_representations'], tokenized_list[1])
