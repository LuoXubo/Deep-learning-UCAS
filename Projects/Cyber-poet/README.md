# Introduction

This project aims to create a poetry generator using a recurrent neural network (RNN) with long short-term memory (LSTM) units. The model is trained on a dataset of poems and generates new poems based on the patterns it has learned.

# Dataset

The "tang.npz" is a preprocessed dataset of 57,580 Tang poems. Each poem is limited to 125 characters. The dataset is saved in npz format and contains three parts:

- data: a numpy array of shape (57580, 125) representing the poems. Each poem is represented as a sequence of 125 characters. Characters that are not in the poem are padded with spaces. The characters are converted to their index in the vocabulary.

- ix2word: a mapping from index to character.

- word2ix: a mapping from character to index.

# Model

The model is a character-level RNN with LSTM units. The input to the model is a sequence of characters, and the output is a sequence of characters. The model is trained to predict the next character in the sequence given the previous characters. The model is trained using the Adam optimizer and the categorical cross-entropy loss function.

# Training

The model is trained on the "tang.npz" dataset for 50 epochs with a batch size of 128. The model is saved after each epoch, and the best model is selected based on the validation loss. The model is evaluated on a validation set of 10% of the data.

# Generation

The model is used to generate new poems by sampling characters from the model's output. The model is given a seed text to start the generation process, and it generates a sequence of characters based on the seed text. The model continues generating characters until it reaches a predefined length or a stop token.

# Results

The model generates poems that are similar in style to the training data. The poems have a similar structure and use similar words and phrases. The model is able to generate poems that are coherent and have a consistent theme. The model is able to generate poems that are similar to the training data but are not direct copies of the training data.

# References

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Generating Text with Recurrent Neural Networks](https://www.tensorflow.org/tutorials/text/text_generation)
- [Text Generation With LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)
- [How to Develop a Word-Level Neural Language Model and Use it to Generate Text](https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/)
- [How to Develop a Character-Based Neural Language Model in Keras](https://machinelearningmastery.com/develop-character-based-neural-language-model-keras/)
- [How to Develop a Word-Level Neural Language Model and Use it to Generate Text](https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/)

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
