import numpy as np
import pandas as pd

import tensorflow as tf
from keras import backend as K
from tensorflow.keras import layers , activations , models , preprocessing, utils

import pickle
import glob

def test_build(csvFilePath, batch, epoch, savepath):
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
    encoder_model.save(savepath+'Encoder-'+str(maxlen_questions)+'-'+str(maxlen_answers)+'.h5')
    decoder_model.save(savepath+'Decoder-'+str(maxlen_questions)+'-'+str(maxlen_answers)+'.h5')
    with open(savepath+'tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer, encoder_model, decoder_model, maxlen_questions, maxlen_answers

#-------------------------------------------------------------------------------------------------------------------------------------#

#test_build('../../../../common.csv', 16, 5, '../../../../bots/JDW9W2YHR9FJE80FC/')
#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#

# Inference Prediction
def preprocess_input(input_sentence,tokenizer,maxlen_questions):
    tokens = input_sentence.lower().split()
    tokens_list = []
    for word in tokens:
        tokens_list.append(tokenizer.word_index[word]) 
    return preprocessing.sequence.pad_sequences([tokens_list] , maxlen=maxlen_questions , padding='post')

def responce_chatbot(botID,question):
    model_location = f'../../../../bots/{botID}/'
    models = sorted(glob.glob(f'../../../../bots/{botID}/*h5'))
    maxlen_questions = models[0].split('Decoder')[-1].split('-')[1]
    maxlen_answers = models[0].split('Decoder')[-1].split('-')[-1].split('.')[0]
    encoderModel = tf.keras.models.load_model(models[1])
    decoderModel = tf.keras.models.load_model(models[0])
    with open(f'../../../../bots/{botID}/tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    # print(encoderModel, decoderModel, tokenizer)

    states_values = encoderModel.predict(preprocess_input(question,tokenizer,int(maxlen_questions)), verbose=0)
    empty_target_seq = np.zeros((1 , 1))
    empty_target_seq[0, 0] = tokenizer.word_index['start']
    stop_condition = False
    decoded_translation = ''
    
    while not stop_condition :
        dec_outputs , h , c = decoderModel.predict([empty_target_seq] + states_values, verbose=0)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        
        for word , index in tokenizer.word_index.items() :
            if sampled_word_index == index :
                decoded_translation += f' {word}'
                sampled_word = word
        
        if sampled_word == 'end' or len(decoded_translation.split()) > int(maxlen_answers):
            stop_condition = True
            
        empty_target_seq = np.zeros((1 , 1))  
        empty_target_seq[0 , 0] = sampled_word_index
        states_values = [h , c] 
    print(f'Human: {question}')
    print()
    decoded_translation = decoded_translation.split(' end')[0]
    print(f'Bot: {decoded_translation}')
    print('-'*25)
        

#-------------------------------------------------------------------------------------------------------------------------------------#


responce_chatbot('JDW9W2YHR9FJE80FC','you can not move')
#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#