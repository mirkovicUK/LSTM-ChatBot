LSTM Bot Udacity nanodegree 

This project is greatly inpacted by folowing PyTorch tutorials:
https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html 


Project Overview
In this project, you will build a chatbot that can converse with you at the command line.
The chatbot will use a Sequence to Sequence text generation architecture with an LSTM as
it's memory unit. You will also learn to use pretrained word embeddings to improve the
performance of the model. At the conclusion of the project, you will be able to show 
your chatbot to potential employers.

Additionally, you have the option to use pretrained word embeddings in your model. 

A sequence to sequence model (Seq2Seq) has two components:

An Encoder consisting of an embedding layer and LSTM unit.
A Decoder consisting of an embedding layer, LSTM unit, and linear output 
unit.
The Seq2Seq model works by accepting an input into the Encoder, passing 
the hidden state from the Encoder to the Decoder, which the Decoder uses 
to output a series of token predictions.

Please choose a dataset from the Torchtext website. We recommend looking 
at the Squad dataset first. Here is a link to the website where you can 
view your options:

https://pytorch.org/text/stable/datasets.html