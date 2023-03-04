import chatNeuron as cn
from tensorflow.keras import layers , activations , models , preprocessing, utils
import numpy as np
import tensorflow as tf
import pickle

SAVE = '../'

tokenizer, encoder_model, decoder_model, maxlen_questions, maxlen_answers = cn.build_AI_Chatbot('../common.csv', 16, 100, SAVE)

# Loading Model and Weights
encoder_model = tf.keras.models.load_model(SAVE+'Encoder.h5')
decoder_model = tf.keras.models.load_model(SAVE+'Decoder.h5')
with open(SAVE+'tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def preprocess_input(input_sentence):
        tokens = input_sentence.lower().split()
        tokens_list = []
        for word in tokens:
            tokens_list.append(tokenizer.word_index[word]) 
        return preprocessing.sequence.pad_sequences([tokens_list] , maxlen=maxlen_questions , padding='post')

tests = ['You can not move', 'ARE YOU A FOOTBALL', 'Stupid', 'you are idiot', 'what is greenpeace']
for i in range(5):
    states_values = encoder_model.predict(preprocess_input(tests[i]), verbose=0)
    empty_target_seq = np.zeros((1 , 1))
    empty_target_seq[0, 0] = tokenizer.word_index['start']
    stop_condition = False
    decoded_translation = ''
    
    while not stop_condition :
        dec_outputs , h , c = decoder_model.predict([empty_target_seq] + states_values, verbose=0)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        
        for word , index in tokenizer.word_index.items() :
            if sampled_word_index == index :
                decoded_translation += f' {word}'
                sampled_word = word
        
        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True
            
        empty_target_seq = np.zeros((1 , 1))  
        empty_target_seq[0 , 0] = sampled_word_index
        states_values = [h , c] 
    print(f'Human: {tests[i]}')
    print()
    decoded_translation = decoded_translation.split(' end')[0]
    print(f'Bot: {decoded_translation}')
    print('-'*25)