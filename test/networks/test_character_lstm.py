from chatbot.src.networks.character_lstm import CharacterLSTM
from learning_framework.src.train.preprocess.character_embedding_preprocess import CharacterEmbeddingPreprocess
from learning_framework.src.train.sample.text_sample import TextSample
import torch


def test_character_lstm():
    text_to_embed = ['The long winter was very scary at first, but later we discovered that the snow can be turned to water.']
    embedding = CharacterEmbeddingPreprocess()(TextSample(text_to_embed)).text_data
    text_tensor = torch.from_numpy(embedding)[:, None, :]
    network = CharacterLSTM(initial_embedding_size=256, hidden_layer_size=2048)
    output = network(text_tensor)
