# The Main Ideas Behind Transformers and Attention
link: https://learn.deeplearning.ai/courses/attention-in-transformers-concepts-and-code-in-pytorch/lesson/ym7dj/the-main-ideas-behind-transformers-and-attention

- The LLMs are fundamentally based on transformers

Transformers can look complicated, but, fundamentally they require 3 main parts:
- Attention
- Positional Encoding
- Word Embedding

- The first part, **Word Embedding** converts words, bits of words and symbols, collectively called Tokens, into numbers. We need this because Transformers are a type of Neural network, and neural networks only have numbers for input values
- Positional Encoding helps keep track of word order
- Transformer establishes relationships among words with Attention. There are different kinds of Attention. Here are some common types of Attention:
    - **Self Attention** works by seeing how similar each word to all of the words in the sentence, including itself. Once the similarities are calculated, they are used to determine how the Transformer encodes each word.
    - ****



