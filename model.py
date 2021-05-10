import torch
import torch.nn as nn
import torch.nn.functional as F


class NABoE(nn.Module):
    def __init__(self, embedding_word, embedding_entity, number_of_classes, dropout_probability, word_u):
        super(NABoE, self).__init__()

        self.word_u = word_u
        self.embedding_word = nn.Embedding(embedding_word.shape[0], embedding_word.shape[1], padding_idx=0)
        self.embedding_word.weight = nn.Parameter(torch.FloatTensor(embedding_word))
        self.embedding_entity = nn.Embedding(embedding_entity.shape[0], embedding_entity.shape[1], padding_idx=0)
        self.embedding_entity.weight = nn.Parameter(torch.FloatTensor(embedding_entity))
        self.attention_layer = nn.Linear(2, 1)
        self.dropout = nn.Dropout(p=dropout_probability)
        self.output_layer = nn.Linear(embedding_word.shape[1], number_of_classes)

    def forward(self, word_ids, entity_ids, prior_probs):
        sum_words_vector = self.embedding_word(word_ids).sum(1)
        vectors_entity = self.embedding_entity(entity_ids)
        div_norm = torch.norm(sum_words_vector, dim=1, keepdim=True).clamp(min=1e-12).detach()
        words_n_vector = sum_words_vector / div_norm
        div_norm_2 = torch.norm(vectors_entity, dim=2, keepdim=True).clamp(min=1e-12).detach()
        entity_n_vector = vectors_entity / div_norm_2
        words_n_vector = words_n_vector.unsqueeze(1)
        cosin_similarity = (words_n_vector * entity_n_vector).sum(2, keepdim=True)
        prior_p = prior_probs.unsqueeze(2)
        attention_features = torch.cat((prior_p, cosin_similarity), 2)
        attention_logit = self.attention_layer(attention_features).squeeze(-1)
        attention_logit = attention_logit.masked_fill_(entity_ids == 0, -1e32)
        attention_weights = F.softmax(attention_logit, dim=1)
        atten_weight_unsqueeze = attention_weights.unsqueeze(-1)
        vector_features = (vectors_entity * atten_weight_unsqueeze).sum(1)
        if self.word_u:
            div_sum = (word_ids != 0).sum(dim=1, keepdim=True).type_as(sum_words_vector)
            word_vector_features = sum_words_vector / div_sum
            vector_features = vector_features + word_vector_features

        vector_features = self.dropout(vector_features)
        return self.output_layer(vector_features)

