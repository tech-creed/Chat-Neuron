import numpy as np
import pandas as pd

import tensorflow as tf
from keras import backend as K
from tensorflow.keras import layers , activations , models , preprocessing, utils

import pickle

def build_AI_Chatbot(csvFilePath, batch, epoch, savepath):
    # dataset for the AI chatbot
    dataset = pd.read_csv(csvFilePath)
    questions = dataset['questions']
    answers = dataset['answers']

#-------------------------------------------------------------------------------------------------------------------------------------#

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
    VOCAB_SIZE = len(tokenizer.word_index)+1

#-------------------------------------------------------------------------------------------------------------------------------------#

    # encoder_input_data
    tokenized_questions = tokenizer.texts_to_sequences(questions)
    maxlen_questions = max([len(x) for x in tokenized_questions])
    padded_questions = preprocessing.sequence.pad_sequences(tokenized_questions , maxlen=maxlen_questions , padding='post')
    encoder_input_data = np.array(padded_questions)

    # decoder_input_data
    tokenized_answers = tokenizer.texts_to_sequences(answers)
    maxlen_answers = max([len(x) for x in tokenized_answers])
    padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers , maxlen=maxlen_answers , padding='post')
    decoder_input_data = np.array(padded_answers)

    # decoder_output_data
    tokenized_answers = tokenizer.texts_to_sequences(answers)
    for i in range(len(tokenized_answers)) :
        tokenized_answers[i] = tokenized_answers[i][1:]
    padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers , maxlen=maxlen_answers , padding='post')
    onehot_answers = utils.to_categorical(padded_answers , VOCAB_SIZE)
    decoder_output_data = np.array(onehot_answers)

#-------------------------------------------------------------------------------------------------------------------------------------#

    # Embedding, LSTM and Desne layers
    encoder_inputs = tf.keras.layers.Input(shape=(maxlen_questions ,))
    encoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 200 , mask_zero=True) (encoder_inputs)
    encoder_outputs , state_h , state_c = tf.keras.layers.LSTM(200 , return_state=True)(encoder_embedding)
    encoder_states = [ state_h , state_c ]

    decoder_inputs = tf.keras.layers.Input(shape=(maxlen_answers , ))
    decoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, 200 , mask_zero=True) (decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(200 , return_state=True , return_sequences=True)
    decoder_outputs , _ , _ = decoder_lstm (decoder_embedding , initial_state=encoder_states)


    decoder_dense = tf.keras.layers.Dense(VOCAB_SIZE , activation=tf.keras.activations.softmax) 
    output = decoder_dense (decoder_outputs)

#-------------------------------------------------------------------------------------------------------------------------------------#

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    #model.summary()

    model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=batch, epochs=epoch) 

#-------------------------------------------------------------------------------------------------------------------------------------#
    
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=(200 ,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(200 ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)
    
    # Saving the Model and Tokenizer
    encoder_model.save(savepath+'Encoder.h5')
    decoder_model.save(savepath+'Decoder.h5')
    with open(savepath+'tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer, encoder_model, decoder_model, maxlen_questions, maxlen_answers

#-------------------------------------------------------------------------------------------------------------------------------------#