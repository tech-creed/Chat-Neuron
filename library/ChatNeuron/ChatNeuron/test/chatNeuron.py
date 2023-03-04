import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers , activations , models , preprocessing, utils

def dataPreparation(csvFilePath):
    # dataset for the AI chatbot
    dataset = pd.read_csv(csvFilePath)
    questions = dataset['questions']
    answers = dataset['answers']

    return questions, answers

def gstTokenizer(questions,answers):
    # tokenizer from the dataset
    answers_with_tags = []
    for i in range(len(answers)):
        if type(answers[i]) == str:
            answers_with_tags.append(answers[i])
        else:
            questions.pop(i)

    answers = []
    for i in range(len(answers_with_tags)) :
        answers.append('<START> ' + answers_with_tags[i] + ' <END>')

    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(questions + answers)
    vocab_len = len(tokenizer.word_index)+1

    return vocab_len, tokenizer

def Tokenize(questions,answers,tokenizer,vocab_len):
    # encoder_input_data
    tokenized_questions = tokenizer.texts_to_sequences(questions)
    maxlen_questions = max([len(x) for x in tokenized_questions])
    padded_questions = preprocessing.sequence.pad_sequences(tokenized_questions , maxlen=maxlen_questions , padding='post')
    encoder_input = np.array(padded_questions)

    # decoder_input_data
    tokenized_answers = tokenizer.texts_to_sequences(answers)
    maxlen_answers = max([len(x) for x in tokenized_answers])
    padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers , maxlen=maxlen_answers , padding='post')
    decoder_input = np.array(padded_answers)

    # decoder_output_data
    tokenized_answers = tokenizer.texts_to_sequences(answers)
    for i in range(len(tokenized_answers)) :
        tokenized_answers[i] = tokenized_answers[i][1:]
    padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers , maxlen=maxlen_answers , padding='post')
    onehot_answers = utils.to_categorical(padded_answers , vocab_len)
    decoder_output = np.array(onehot_answers)

    return encoder_input, decoder_input, decoder_output