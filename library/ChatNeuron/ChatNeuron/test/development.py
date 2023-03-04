import chatNeuron as cn

questions, answers  = cn.dataPreparation('../csvTest.csv')
vocab_len, tokenizer = cn.gstTokenizer(questions, answers)
encoderInput, decoderInput, decoderOutput = cn.Tokenize(questions, answers, tokenizer, vocab_len)

print(encoderInput, decoderInput, decoderOutput)

