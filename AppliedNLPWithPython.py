import tensorflow as tf
import keras
import theano as T
import nltk.tokenzie
import sklearn.feature_extraction.text
import gensim
import PDFMiner
import numpy as np

#Chapter 1 - What is NLP?
#Creating weights and biases dictionaries.
weights = {'input': tf.Variable(tf.random_normal([state_size+1, state_size])), 'output': tf.Variable(tf.random_normal([state_size, n_classes]))}
biases = {'input': tf.Variable(tf.random_normal([1, state,_size])), 'output': tf.Variable(tf.random_normal([1, n_classes]))}

#Defining placeholders and variables.
X = tf.placeholder(tf.float32, [batch_size, bprop_len])
Y = tf.placeholder(tf.int32 [batch_size, bprop_len])
init_state = tf.placeholder(tf.float32, [batch_size, state_size])
input_series = tf.unstack(X, axis=1)
labels = tf.unstack(Y, axis=1)
current_state = init_state
hidden_states []

#Passing values from one hidden state to the next.
for input in input_series: #Evaluating each input within the series of inputs.
    input = tf.reshape(input, [batch_size, 1]) #Reshaping input into MxN tensor.
    input_state = tf.concat([input, current_state], axis=1)
    #Concatenating input and current state tensors.
    _hidden_state = tf.tanh(tf.add(tf.matmul(input_state, weights['input']), biases['input'])) #Tanh transformation.
    current_state = _hidden_state #Updating the current state.
    
#Keras
def create_model():
    model = Sequential()
    model.add(ConvLSTM2D(filters=40, kernal_size=(3, 3), input_shape=(None, 40, 40, 1), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=40, kernal_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=40, kernal_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=40, kernal_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=1, kernal_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))
    model.compile(loss='binary_crossentropy', optimizer='adadelta')
    return model

#Theano
X, Y = T.matrix(), T.vector(dtype=theano.config.floatX)
weights = init_weights(weight_shape)
biases = init_biases(bias_shape)
predicted_y = T.argmax(model(X, weights, bases), axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(predicted_y, Y))
gradient = T.grad(cost=cost, wrt=weights)
update = [[weights, weights = gradient * 0.05]]
train = theano.function(inputs_[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=predicted_y, allow_input_downcast=True)

for i in range (0, 10):
    print(predict(test_x_data)[i:i+1]))
    
if __name__ == '__main__':
    
    model_predict()
    
#Chapter 2 - Review of Deep Learning
#MLP model.
def mlp_model(train_data=train_data, learning_rate=0.01, iters=100, num_hidden=256):
    weights = {'input': tf.Variable(tf.random_normal([train_x.shape[1], num_hidden])), 'hidden1': tf.Variable(tf.random_normal([num_hidden])), 'output': tf.Variable(tf.random_normal([num_hidden, 1]))}
    biases = {'input': tf.Variable(tf.random_normal([num_hidden])), 'hidden1': tf.Variable(tf.random_normal([num_hidden])), 'ouput': tf.Variable(tf.random_normal([1]))}
    
#Creating training and test sets
train_x, train_y = train_data[0:int(len(train_data)*.67, 1:train_data.shape[1]], train_data[0:int(len(train_data)*.67), 0]
test_x, test_y = train_data[int(len(train_data)*.67):, 1:train_data.shape[1]], train_data[int(len(train_data)*.067: 0]
                                                                                              
#Creating placeholder values and instantiating weights and biases as dictionaries
x = tf.placeholder('float', shape = (None, 7))
y = tf.placeholder('float', shape = (None, 1))
                                                                                
#Passing data through input, hidden and output layers.
input_layer = tf.add(tf.matmul(X, weights['input']), biases['input'])
input_layer = tf.nn.sigmoid(input_layer)
input_layer = tf.nn.droput(input_layer, 0.20)
                                                                                              
hidden_layer = tf.add(tf.multiply(input_layer, weights['hidden1']), biases['hidden1'])
hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.dropout(hidden_layer, 0.20)
                                                                                              
output_layer = tf.add(tf.multiply(hidden_layer, weights['output']), biases['output'])
                                                                                              
#RNN model.
def build_rnn(learning_rate=0.02, epochs=100, state_size=4):
    #Loading data.
    x, y = load_data(); scaler = MinMaxScaler(feature_range=(0, 1))
    x, y = scaler.fit_transform(x), scaler.fit_transform(y)
    train_x, train_y = x[0:int(math.floor(len(x)*.67)), :], y[0:int(math.floor(len(y)*.67))]
                                                                                              
    #Creating weights and biases dictionaries.
    weights = {'input': tf.Variable(tf.random_normal([state_size+1, state_size])), 'output': tf.Variable(tf.random_normal([state_size, train_y.shape[1]]))}
    biase= {'input': tf.Variable(tf.random_normal([1, state_size])), 'output': tf.Variable(tf.random_normal([1, train_y.shape[1]]))}
                                                                                        
#Defining placeholders and variables.
    X = tf.placeholder(tf.float32, [batch_size, train_x.shape[1]])
    Y = tf.placeholder(tf.int32, [batch_size, train_x.shape[1]])
    init_state = tf.placeholder(tf.float32, [batch_size, state_size])
    input_series = tf.unstack(X, axis=1)
    current_state = init_state
    hidden_states = []
    
#Passing values from one hidden state to the next
for input in input_series: #Evaluating each input within the series of inputs.
input = tf.reshape(input, [batch_size, 1]) #Reshaping input into MxN tensor.
input_state = tf.concat([input, current_state], axis=1)
#Concatenating input and current state tensors.
_hidden_state = tf.tanh(tf.add(tf.matmul(input_state, weights['input']), biases['input'])) #Tanh transformation.
hidden.states.append(_hidden_state) #Appending the next state.
current_state = _hidden_state #Updating the current state.

logits = [tf.add(tf.matmul(state, weights['output']), biases['output']) for state in hidden_states]

#Predictions for each logit within the series.
predicted_labels = [tf.nn.softmax(logit) for logit in logits]

#LSTM Model
X = tf.placeholder(tf.float32, (None, None, train_x.shape[1]))
Y = tf.placeholder(tf.float32, (None, train_y.shape[1]))
weights = {'output': tf.Variable(tf.random_normal([n_hidden, train_y.shape[1]]))}
biases = {'output': tf.Variable(tf.random_normal([train_y.shape[1]]))}
input_series = tf.reshape(X, [-1, train_x.shape[1]])
input_series = tf.split(input_series, train_x.shape[1], 1)

lstm = rnn.core_rnn_cell.BasicLSTMCell(num_units=n_hidden, forget_bias=1.0, reuse=None, state_is_tuple=True)
_outputs, states = rnn.static_rnn(lstm, input_series, dtype=tf.float32)
predictions = tf.add(tf.matmul(_outputs[-1], weights['output'], biases['output'])
accuracy = tf.reduce_mean(tf.cast(tf.equal)tf.argmax(tf.nn.softmax(predictions), 1)tf.argmax(Y, 1)), dtype=tf.float32)),
error = tf.reduce_mean(tf.nn,softmax_cross_entropy_with_logits(labels=Y, logits=predictions))
adam_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(error)
                                                                                          
#Chapter 3 - Working with Raw Text.
#Sample text.
sample_text = "I am a student from the University of Alabama. I was born in Ontario, Canada and I am a huge fan of the United States. I am going to get a degree in Philosophy to improve my chances of becoming a Philosophy professor. I have been working towards this goal for 4 years. I am currently enrolled in a PhD program. It is very difficult but I am confident that it will be a good decision."

#Tokenizing the sample data.
from nltk.tokenize import word_tokenzie, sent_tokenzie
sample_word_tokens = word_tokenzie(sample_text)
sample_word_tokens = sent_tokenzie(sample_text)

#RegexpTokenizer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
sample_word_tokens = tokenizer.tokenzie(str(sample_word_tokens))
sample_word_tokens = [word.lower() for word in sample_word_tokens]

#Bag-of-Words Model
def bag_of_words(text):
    _bag_of_words = [collections.Counter(re.findall(r'\w+', word)) for word in text]
    bag_of_words = sum(_bag_of_words, collections.Counter())
    return bag_of_words

#CounterVectorizer
from sklearn.feature_extraction.text import CountVectorizer
def bow_sklearn(text=sample_sent_tokens):
    c = CountVectorizer(stop_words='english', token_pattern=r'\w+')
    converted_data = c.fit_transform(text).todense()
    print(converted_data.shape)
    return converted_data, c.get_feature_names()
    
#Spam Detection.
#Fitting training algorithm.
1 = LogisticRegression(penalty='l1')
accuracy_scores, auc_scores = [], []

#Collecting distribution of accurancy and AUC scores.
for i in range(trials):
    if i%10 == 0 and i > 0:
        print('Trial ' + str(i) + ' out of 100 completed')
    l.fit(train_x, train-y)
    predicted_y_values = l.predict(train_x)
    accuracy_scores.append(accuracy_score(train_y, predicted_y_values))
    fpr, tpr = roc_curve(train_y, predicted_y_values)[0], roc_curve(train_y, predicted_y_values) [1]
    auc_scores.append(auc(fpr, tpr))
    
#Term Frequency Inverse Document Frequency.
document_list = list([sample_text, text])

#Calculating TFIDF
text = "I was a student at the University of Pennsylvania, but now work on Wall Street as a lawyer. I have been living in New York for roughly five years now, however, I am looking forward to eventually retiring to Texas once I have saved up enough money to do so."
document_list = list([sample_text, text])

#tf_idf_example() function.
def tf_idf_example(textblobs=[text, text2]):
def term_frequency(word, textblob): (1)
return textblob.words.count(word)/float(len(textblob.words))

def document_counter(word, text):
return sum(1 for blob in text if word in blob)

def idf(word, text): (2)
return np.log(len(text) /1 + float(document_counter(word, text)))

def tf_idf(word, blob, text):
return term_frequency(word, blob) * idf(word, text)

output = list()
for i, blob in enumerate(textblobs):
output.append({word: tf_idf(word, blob, textblobs) for word in blob.words})
print(output)

#Classifying Movie Reviews.
#TfidfVectorizer() method.
def remove_non_ascii(text):
    return ".join([word for word in text if ord(word) < 128])
    
#Loading data.
def load_data():
    negative_review_strings = os.listdir#(File name)
    positive_review_strings = os.listdir#(File name)
    negative_reviews, positive_reviews = [], []
    
#Load file with open() function.
for positive_review in positive_review_strings:
    with open(#File name'+str(positive_review), 'r') as positive_file: positive_reviews.append(remove_non_ascii(positive_file.read()))
    
for negative_review in positive_review_strings:
    with open(#File name'+str(negative_review), 'r') as negative_file: negative_reviews.append(remove_non_ascii(negative_file.read()))
    
#Load and preprocess text data.
x, y = load_data()
t = TfidVectorizer(min_df=10, max_df=300m, stop_words='english', token_pattern=r'\w+')
x = t.fit_transform(x).todense()

#Performing weight regularization.
regularization = tf.contrib.layers.l2_regularizer(scale=0.0005, scope=None)
regularization_penalty = tf.contrib.layers.apply_regularization(regularization, weights.values())
cross_entropy = cross_entropy + regularization_penalty
        
#Chapter 4 - Topic Modelling and Word Embeddings
#Example of code utlized to first create out topic model.
def create_topic_model(model, n_topics=10, max_inter=5, min_df=10, max_df=300, stop_words='english', token_pattern=r'\w+'):
    print(model + ' topic modeL \n')
    data = load_data()[0]
    if model == 'tf':
        feature_extractor = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words, token_pattern=r'\w+')
    else:
        feature_extractor = TfidVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words, token_pattern=r'\w+')
    processed_data = feature.extractor.fit_transform(data)
    
#LDA Model.
lda_model = LatentDirichletAllocation(n_topics=n_topics, learning_method='online', learning_offset=50., max_iter=max_iter, verbose=1)
lda_model.fit(processed_data)
tf_features = feature_extractor.get_feature_names()
print_topics(model=lda_model, feature_names=tf_features, n_top_words=n_top_words)

#Utilizing Gensim package.
def gensim_topic_model():
    def remove_stop_words(text): (1)
    word_tokens = word_tokenzie(text.lower())
    word_tokens = [word for word in word_tokens if word not in stop_words and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', word)]
    return word_tokens
    data = load_data() [0]
    cleaned_data = [remove_stop_words(data[i]) for i in range (0, len(data))]

#Removing words that appear too frequently or not frequently enough.
dictionary = gensim.corpora.Dictionary(cleaned_data)
dictionary.filter_extremes(no_below=100, no_above=300)
corpus = [dictionary.doc2bow(text) for text in cleaned_data]
lda_model = models.LdaModel(corpus=corpus, num_topics=n_topics, ida2word=dictionary, verbose=1)

#Non-Negative Matrix Factoriization.
def nmf_topic_model():
    def create_topic_model(model, n_topics=10, max_iter-5, min_df=10, max_df=300, stop_words='english', token_pattern=r'\w+'):
        print(model + ' NMF topic model: ')
        data = load_data()[0]
        if model == 'tf'
        feature_extractor = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words, token_pattern=token_pattern)
        else
        feature extractor = TfidVectorizer(min_df=min_df, max_df=max_df, stop_words=stopwords, token_pattern=token_pattern)
        processed_data = feature_extractor.fit_transform(data)
        nmf_model = NMF(n_components=n_components, max_iter=max_iter)
        nmf_model.fit(processed_data)
        tf_features = feature_extractor.get_feature_names()
        print_topics(model=nmf_model, feature_names=tf_features, n_top_words=n_topics)
        
    create_topic_model(model='tf')
    
#Training a Word Embedding (Skip-Gram).
#Beginning implementation of TensorFlow Word2Vec Skip-Gram model.
def remove non_ascii(text):
    return ".join([word for word in text if ord(word) < 128])

def load_data(max_pages=100):
    return_string = StringIO()
    device = TextConverter(PDFResourceManager(), return_string, codec='utf-8', laparams=LAParams())
    interpreter = PDFPageInterpreter(PDFResourceManager(), device=device)
    filepath = file(#File name, 'rb')
    for page in PDFPage.get_pages(filepath, set(), maxpages=max_pages, caching=True, check_extractable=True): interpreter.process_page(page)
    text_data = return_string.getvalue()
    filepath.close(), device.close(), return_string.close()
    return remove_non_ascii(text_data)

#Utilizing PDFMiner module.
def gensim_preprocess_data():
    data = load_data()
    sentences = sent_tokenize(data)
    tokenized_sentences = list([word_tokenize(sentence) for sentence in sentences])
    for i in range (0, len(tokenized_sentences)):
        tokenized_sentences[i] = [word for word in tokenized_sentences[i] if word not in punctuation]
    return tokenized_senteces
    
#Gensim implementation of Skip-Gram model.
def gensim_skip_gram():
    sentences = gensim_preprocess_data()
    skip_gram - Word2Vec(sentences=sentences, window=1, min_count=10, sg=1)
    word_embedding = skip_gram[skip_gram.wv.vocab] (1)
    
    #Visualizing words as vectors.
    pca = PCA(n_components=2)
    word_embedding = pca.fit_transform(word_embedding)
    #Plotting results from trained word embedding.
    plt.scatter(word_embedding[:, 0], word_embedding[:, 1])
    word_list = list(skip_gram.wv)
    for i, word in enumerate(word_list):
        plt.annotate(word, xy=(word_embedding[i, 0], word_embedding[i, 1]))
        
#Implementing word embedding in TensorFlow.
def tf_preprocess_data(window_size=window_size):
    def one_hot_encoder(index, vocab_size):
        vector = np.zeros(vocab_size)
        vector[index] = 1
        return vector
        
    text_data = load_data()
    vocab_size = len(word_tokenize(text_data))
    word_dictionary = {}
    for index, word in enumerate(word_tokenize(text_data)):
        word_dictionary[word] = index
        
    sentences = sent_tokenize(text_data)
    tokenized_sentences = list([word_tokenize(sentence) for sentence in sentences])
    n_gram_data = []
    
#Creating word pairs for word2vec model
for sentence in tokenized_sentences:
    for index, word in enumerate(sentence):
        if word not in punctuation:
            for_word in sentence[max(index - window_size, 0): min(index + window_size, len(sentence)) + 1]:
                if_word != word:
                    n_gram_data.append([word, _word])

#One-hot encoding data and creating dataset intrepretable by Skip-Gram model.
x, y = np.zeros([len(n_gram_data), vocab_size]), np.zeros([len(n_gram_data), vocab_size])
for i in range(0, len(n_gram_data)):
    x[i, :] = one_hot_encoder(word_dictionary[n_gram_data[i][0]], vocab_size=vocab_size)
    y[i, :] = one_hot_encoder(word_dictionary[n_gram_data[i][1]], vocab_size=vocab_size)
return x, y, vocab_size, word_dictionary

#Function to construct Skip-Gram model.
def tensorflow_word_embedding(learning_rate=learning_rate, embedding_dim=embedding_dim):
    x, y, vocab_size, word_dictionary = tf_preprocess_data()
    
    #Defining TensorFlow variables and placeholder.
    X = tf.placeholder(tf.float32, shape=(None, vocab_size))
    Y = tf.placeholder(tf.float32, shape=(None, vocab_size))
    
    weights = {'hidden': tf.Variable(tf.random_normal([vocab_size, embedding_dim])), 'output': tf.Variable(tf.random_normal([embedding_dim, vocab_size]))}
    biases = {'hidden': tf.Variable(tf.random_normal([vocab_size, embedding_dim])), 'output': tf.Variable(tf.random_normal([vocab_size]))}
    input_layer = tf.add(tf.matmul(X, weights['hidden']), biases['hidden'])
    output_layer = tf.add(tf.matmul(input_layer, weights['output']), biases['output'])
    
#Global Vectors for Word Representation.
def load_embedding(embedding_path=#'Embedding path'): vocubulary, embedding = [], []
for line in open(embedding_path, 'rb').readlines():
    row = line.strip().split(' ')
    vocabulary.append(row[0]), embedding.append(row[1:])
    vocabulary_length, embedding_dim, len(vocabulary), len(embedding[0])
    return vocabular, np.asmatrix(embedding), vocabulary_length, embedding_dim
    
#Using Trained Word Embeddings with LSTMs.
#Paragraph for additional training data for word embedding.
sample_text = "'Living in different places has been the greatest experience that I have had in my life. It has allowed me to understand people from different walks of life, as well as to question some of my own biases I have had with respect to people who did not grow up as I did. If possible, everyone should take an opportunity to travel someone seperate from where they grew up."'.replace('\n', ")

def sample_text_dictionary(data=_sample_text):
    count, dictionary = collections.Counter(data).most_common(), {} #creates list of word/count pairs;
    for word, _ in count:
        dictionary[word] = len(dictionary) #len(dictionary)
        increases each iteration
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    dictionary_list = sorted(dictionary.items(),
    key = lamba x : x [1])
    return dictionary, reverse_dictionary, dictionary_list
    
#Paragraph2Vec Example with Movie Data Review.
#Preprocessing data.
def gensim_preprocess_data(max_pages):
    sentences = namedtuple('sentence', 'words tags')
    _sentences = sent_tokenize(load_data(max_pages=max_pages))
    documents = []
    for i, text in enumerate(_sentences):
        words, tags = text.lower().split(), [i]
        documents.append(sentences(words, tags))
    return documents

#Defining sample documents.
sample_text1 = "'I love italian food. My favourite items are pizza and pasta, especially garlic bread. The best italian food I have had has been in New York. Little Italy was very fun"'
sample_text2 = "'My favourite time of italian food is pasta with alfredo sauce. It is very creamy but the cheese is the best part. Whenevr I go to an Italian restaurant, I am always certain to get a plate."'
        
#Chapter 5 - Text Generation, Machine Translation and Other Recurrent Language Modelling Tasks.
#Text Generation with LSTMs.
def preprocess_data(sequence_length, max_pages=max_pages, pdf_file=pdf_file):
    text_data = load_data(max_pages=max_pages, pdf_file=pdf_file)
    characters = list(set(text_data.lower()))
    character_dict = dict(character, i) for i, character in enumerate(characters)
    int_dictionary = dict((i, character) for i, character in enumerate(characters))
    num_chars, vocab_size = len(text_data), len(characters)
    x, y = [], []
    
    for i in range(0, num_chars - sequence_length, 1):
        input_sequence = text_data[i: i+sequence_length]
        output_sequence = text_data[i+sequence_length]
        x.append([character_dict[character.lower()] for character in input_sequence])
        y.append(character_dict[output_sequence.lower()])
        
    for k in range(0, len(x)): x[i] = [_x for _x in x[i]]
    x = np.reshape(x, (len(x), sequence_length, 1))
    x, y = x/float(vocab_size), np.utils.to_categorical(y)
    return x, y, num_chars, vocab_size, character_dict, int_dictionary

#Creating RNN.    
def create_rnn(num_units=num_units, activation=activation):
    model = Sequential()
    model.add(LSTM(num_units, activation=activation, input_shape=(None, x.shape[1])))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model
    
#Building BRNN.
def create_lstm(input_shape=(1, x.shape[1])):
    model = Sequential()
    model.add(Bidirectional(LSTM(unites=n_units, activation=activation), input_shape=input_shape))
    model.add(Dense(train_y.shape[1]), activation=out_act)
    model.compile(loss='categorial_crossentropy', metrics=['accuracy'])
    return model
    
#Creating a Name Entity Recognition Tagger.
#Turning text data into interpretable format.
def load_data():
    text_data = open('#File name.', 'rb').readlines()
    text_data = [text_data[k].replace('\t', '').split() for k in range(0, len(text_data))]
    index = range(0, len(text_data), 3)
    #Transforming data to matrix format for neural network.
    input_data = list()
    for i in range(1, len(index)-1):
        rows = text_data[index[i-1]:index[1]]
        sentence_no = np.array([i for i in np.repeat(i, len(rows[0]))], dtype=str)
        rows.append(np.array(sentence_no))
        rows = np.array(rows).T
        input_data.append(rows)
        
#Iterating through each line of text file.
text_data[0]
['played', 'on', 'Monday', '(', 'home', 'team', 'in', 'CAPS', ')', ':']
text_data[1]
['VBD', 'IN', 'NPP', '(', 'NN', 'IN', 'NPP', ')', ':']
text_data[2]
['0', '0', '0', '0', '0', '0', '0', '0', '0', '0']

#input_data variable.
input_data = pan.DataFrame(np.concatenate([input_data[j] for j in range(0, len(input_data))]), columns=['word', 'pos', 'tag', 'sent_no'])

labels, vocabulary = list(set(input_data['tag'].values)), list(set(input_data['word'].values))
label_size = len(labels)

aggregate_function = lambda input: [(word, pos, label) for word, post, label in zip(input['word'].values.tolist(), input['pos'].values.tolist(), input['tag'].values.tolist())]

#Transforming words to their integer labels.
x = [[word_dictionary[word[0]] for word in sent] for sent in sentences]
x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
y = [[word_dictionary[word[2]] for word in sent] for sent in sentences]
y = pad_sequences(maxlen=input_shape, sequences=y, padding='post', value=0)
= [np.utils.to_categorical(label, num_classes=label_size) for label in y]

#Training neural network.
def create_brnn():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size+1, output_dim=output_dim, input_length=input_length, mask_zero=True))
    model.add(Bidirectional(LSTM(units=n_units, activation=activation, return_sequences=True)))
    model.add(TimeDistributed(Dense(label_size, activation=out_act)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
    
#Sequence-to-Sequence Models.
#Question and Answer with Neural Network Models.
#Preprocessing data.
dataset = json.load(open(#File name, 'rb'))['data']
questions, answers = [], []
for j in range(0, len(dataset)):
    for k in range(0, len(dataset[j])):
        for i  in range(o, len(dataset[j]['paragraphs'][k]['qas'])):
            questions.append(remove_non_ascii(dataset[j]['paragraphs'][k]['qas'][i]['question']))
            answers.append(remove_non_ascii(dataset[j]['paragraphs'][k]['qas'][i]['answers'][0]['text']))
    
    #Iterating through questions and answers.
    input_chars, output_chars = set(), set()
    
    for i in range(0, len(questions)):
        for char in questions[i]:
            if char not in input_chars: input_chars.add(char.lower())
            
    for i in range(0, len(answers)):
        for char in answers[i]:
            if char not in output_chars: output_chars.add(char.lower())
            
    input_chars, output_chars = sorted(list(input_chars)), sorted(list(output_chars))
    n_encoder_tokens, no_decoder_tokens = len(input_chars), len(output_chars)
    
#Transforming input sequences to one-hot encoded vectors that are interpretable by the neural network.
(code redacted, please see github)
x_encoder = np.zeros((len(questions), max_encoder_len, n_encoder_tokens))
x_encoder = np.zeros((len(questions), max_encoder_len, n_encoder_tokens))
y_encoder = np.zeros((len(questions), max_encoder_len, n_encoder_tokens))

for i, (input, output) in enumerate(zip(questions, answers)):
    for_character, character in enumerate(input):
        x_encoder[i, character, input_dictionary[character.lower()]] = 1.
        
    for_character, character in enumerate(input):
        x_decoder[i, character, input_dictionary[character.lower()]] = 1.
        
        if i > 0: y_decoder[i, _character, output_dictionary[character.lower()]] = 1.
        
#Defining the model
def encoder_decoder(n_encoder_tokens, n_decoder_tokens):
    
    encoder_input = Input(shape=(None, n_encoder_tokens))
    encoder = LSTM(n_units, return_state=True)
    encoder_output, hidden_state, cell_state = encoder(encoder_input)
    encoder_states = [hidden_state, cell_state]
    
    decoder_input = Input(shape=(None, n_decoder_tokens))
    decoder = LSTM(n_units, return_state=True, return_sequences=True)
    decoder_output, _, _ = decoder(decoder_input, initial_state=encoder_states)
    
    decoder = Dense(n_decoder_tokens, activation='softmax')(decoder_output)
    model = Model([encoder_input, decoder_input], decoder)
    model.compile(optimizer='adam', loss='categorical_crosstropy', metrics=['accuracy'])
    model.summary()
    return model