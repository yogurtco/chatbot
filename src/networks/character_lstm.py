import torch.nn as nn
import torch.nn.functional as F
import torch
from learning_framework.src.train.preprocess.character_embedding_preprocess import CharacterEmbeddingPreprocess


class CharacterLSTM(nn.Module):
    def __init__(self, initial_embedding_size: int, hidden_layer_size: int):
        """
        :param initial_embedding_size: initial embedding
        :param hidden_layer_size: hidden layer
        """
        super().__init__()
        self.fc1 = nn.Conv2d(CharacterEmbeddingPreprocess.max_ascii_num, initial_embedding_size, kernel_size=(1, 1))

        self.lstm1_size = (initial_embedding_size, hidden_layer_size)
        self.lstm1 = nn.LSTM(self.lstm1_size[0], self.lstm1_size[1])

        self.lstm2_size = (hidden_layer_size, CharacterEmbeddingPreprocess.max_ascii_num)
        self.lstm2 = nn.LSTM(self.lstm2_size[0], self.lstm2_size[1])

    def forward(self, x):
        x = x.permute(0, 2, 1)[:, :, :, None]
        x = self.fc1(x)
        x = F.relu(x)

        # reshape for LSTM
        x = x[:, :, :, 0].permute(2, 0, 1)
        x, hidden_state = self.lstm1(x)
        x = F.relu(x)
        x, hidden_state = self.lstm2(x)
        return F.log_softmax(x, dim=-1)
