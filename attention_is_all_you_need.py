'''
Reference: https://arxiv.org/abs/1706.03762
'''
import numpy as np
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
# %matplotlib inline
import tensorflow as tf
print('tf version', tf.__version__)
print('built with cuda', tf.test.is_built_with_cuda())
print('gpu available', tf.test.is_gpu_available())

# import tensorflow_datasets as tfds

class tf_module(tf.Module):
    '''
    tf 2.0 requires trainable variables to be class members but I hate python objects so I'll wrap ops in this
    '''
    def __init__(self, **local_vars):
        self.initializing = True
        self.__dict__.update(local_vars)

    def __call__(self, *args):
        # this is called once after init
        if self.initializing:
            self.op = args[0]
            self.initializing = False
            return self
        # subsequent normal calls
        else:
            return self.op(*args)

def make_embeddings(vocab_size, hidden_size):
    embedding_matrix = tf.Variable(tf.random.uniform([vocab_size, hidden_size], -0.05, 0.05), dtype=tf.float32, name='emb')

    @tf_module(**locals())
    def embeddings(inputs):
        # TODO: * tf.sqrt(tf.cast(hidden_size, tf.float32))
        # TODO: share with token projection,
        return tf.nn.embedding_lookup(embedding_matrix, inputs)

    return embeddings

def make_attention_head(hidden_size, num_heads):
    # project in
    Wq = tf.Variable(tf.compat.v2.initializers.glorot_uniform()((hidden_size, hidden_size // num_heads)), dtype=tf.float32, name='Wq')
    Wk = tf.Variable(tf.compat.v2.initializers.glorot_uniform()((hidden_size, hidden_size // num_heads)), dtype=tf.float32, name='Wk')
    Wv = tf.Variable(tf.compat.v2.initializers.glorot_uniform()((hidden_size, hidden_size // num_heads)), dtype=tf.float32, name='Wv')

    @tf_module(**locals())
    def attention_head(q, k, v, mask=None):
        qh = tf.matmul(q, Wq)
        kh = tf.matmul(k, Wk)
        vh = tf.matmul(v, Wv)

        # scaled dot product attn
        logits = tf.matmul(qh, kh, transpose_b=True)
        logits = logits / tf.sqrt(tf.cast(hidden_size, tf.float32))
        if mask is not None:
            logits += (mask * -1e9)
        alphas = tf.nn.softmax(logits)
        c = tf.matmul(alphas, vh)

        return c

    return attention_head

def make_multihead_attention(hidden_size, num_heads):
    attention_heads = [make_attention_head(hidden_size, num_heads) for _ in range(num_heads)]
    Wo = tf.Variable(tf.compat.v2.initializers.glorot_uniform()((hidden_size, hidden_size)), dtype=tf.float32, name='Wo')
    layer_norm = tf.keras.layers.LayerNormalization()

    @tf_module(**locals())
    def multihead_attention(q, k, v, mask=None, keep_prob=0.9):
        outputs = [attention_head(q, k, v, mask) for attention_head in attention_heads]
        output = tf.concat(outputs, axis=-1)
        output = tf.matmul(output, Wo)
        output = tf.nn.dropout(output, rate=1.-keep_prob)
        output = layer_norm(output + q)
        return output
    return multihead_attention

def make_feedforward(hidden_size, ff_size):
    # position-wise ff
    Wff1 = tf.Variable(tf.compat.v2.initializers.glorot_uniform()((hidden_size, ff_size)), dtype=tf.float32, name='Wff1')
    bff1 = tf.Variable(tf.zeros_initializer()((ff_size)), name='bff1')
    Wff2 = tf.Variable(tf.compat.v2.initializers.glorot_uniform()((ff_size, hidden_size)), dtype=tf.float32, name='Wff2')
    bff2 = tf.Variable(tf.zeros_initializer()((hidden_size)), name='bff2')
    layer_norm = tf.keras.layers.LayerNormalization()

    # TODO: pass training=True
    @tf_module(**locals())
    def feedforward(output, keep_prob=0.9):
        ff_output = tf.matmul(output, Wff1) + bff1
        ff_output = tf.nn.relu(ff_output)
        ff_output = tf.matmul(ff_output, Wff2) + bff2
        ff_output = tf.nn.dropout(ff_output, rate=1.-keep_prob)
        output = layer_norm(ff_output + output)
        return output

    return feedforward

def encoder_mask(Xt_batch):
    mask = tf.cast(tf.math.equal(Xt_batch, 0), tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    enc_mask = tf.matmul(mask, mask, transpose_b=True)
    return enc_mask

def make_encoder(num_layers, hidden_size, num_heads, ff_size):
    layers = [(make_multihead_attention(hidden_size, num_heads),
               make_feedforward(hidden_size, ff_size))
              for _ in range(num_layers)]

    @tf_module(**locals())
    def encoder(enc_inputs, enc_mask, keep_prob=0.9):
        enc_outputs = enc_inputs
        for mha, ff in layers:
            enc_outputs = mha(enc_outputs, enc_outputs, enc_outputs, enc_mask, keep_prob)
            enc_outputs = ff(enc_outputs, keep_prob)
        return enc_outputs

    return encoder

def decoder_mask(Yt_batch, max_num_words):
    #TODO: also mask encoder padded outputs
    mask = tf.cast(tf.math.equal(Yt_batch, 0), tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.matmul(mask, mask, transpose_b=True)
    dec_mask = 1 - tf.linalg.band_part(tf.ones((max_num_words, max_num_words)), -1, 0)
    dec_mask = tf.maximum(mask, dec_mask)
    return dec_mask

def make_decoder(num_layers, hidden_size, num_heads, ff_size, tar_vocab_size):
    layers = [(make_multihead_attention(hidden_size, num_heads),
               make_multihead_attention(hidden_size, num_heads),
               make_feedforward(hidden_size, ff_size)) for _ in range(num_layers)]

    W = tf.Variable(tf.compat.v2.initializers.glorot_uniform()((hidden_size, tar_vocab_size)), dtype=tf.float32, name='W')
    b = tf.Variable(tf.zeros_initializer()((tar_vocab_size)), name='b')

    @tf_module(**locals())
    def decoder(enc_outputs, dec_inputs, enc_mask, dec_mask, keep_prob=0.9):
        dec_outputs = dec_inputs
        for dec_mha, enc_mha, ff in layers:
            dec_outputs = dec_mha(dec_outputs, dec_outputs, dec_outputs, dec_mask, keep_prob)
            dec_outputs = enc_mha(dec_outputs, enc_outputs, enc_outputs, enc_mask, keep_prob)
            dec_outputs = ff(dec_outputs, keep_prob)
        logits = tf.matmul(dec_outputs, W) + b
        return logits

    return decoder

def make_loss_fn():
    loss_object = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_fn(logits, labels):
        loss = loss_object(y_true=labels, y_pred=logits)
        loss = tf.reduce_mean(loss)
        return loss

    return loss_fn


def make_train_step(learning_rate, model, loss_fn, loss_metric, acc_metric):
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    ]
    # TODO: warmup steps

    @tf.function(input_signature=train_step_signature)
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            predictions = model(inputs, labels, training=True)
            loss = loss_fn(labels=labels, logits=predictions)
            loss_metric.update_state(loss)
            acc_metric.update_state(labels, tf.argmax(predictions, axis=-1))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return train_step

def make_loss_metric():
    # TODO: unified metrics interface
    loss_metric = tf.keras.metrics.Mean(name='train_loss')
    acc_metric = tf.keras.metrics.Accuracy()
    return loss_metric, acc_metric


class MyTransformerModule(dict, tf.Module):
    def __init__(self, src_vocab_size, tar_vocab_size, max_num_words, hidden_size, num_enc_layers, num_dec_layers, num_heads, ff_size, keep_prob, learning_rate):
        super(MyTransformerModule, self).__init__()

        self.src_embeddings = make_embeddings(src_vocab_size, hidden_size)
        self.tar_embeddings = make_embeddings(tar_vocab_size, hidden_size)
        self.pos_embeddings = make_embeddings(max_num_words, hidden_size)

        self.encoder = make_encoder(num_enc_layers, hidden_size, num_heads, ff_size)
        self.decoder = make_decoder(num_dec_layers, hidden_size, num_heads, ff_size, tar_vocab_size)

        self.loss_fn = make_loss_fn()
        self.loss_metric, self.acc_metric = make_loss_metric()
        self.train_step = make_train_step(learning_rate, self, self.loss_fn, self.loss_metric, self.acc_metric)

        self.max_num_words = max_num_words
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate

    def __call__(self, Xt_batch, Yt_batch, training):
        enc_inputs = tf.nn.dropout(self.src_embeddings(Xt_batch) + self.pos_embeddings(tf.range(self.max_num_words)), rate=1.-self.keep_prob)
        enc_mask = encoder_mask(Xt_batch)
        enc_outputs = self.encoder(enc_inputs, enc_mask)

        dec_inputs = tf.nn.dropout(self.tar_embeddings(Xt_batch) + self.pos_embeddings(tf.range(self.max_num_words)), rate=1.-self.keep_prob)
        dec_mask = decoder_mask(Yt_batch, self.max_num_words)
        logits = self.decoder(enc_outputs, dec_inputs, enc_mask, dec_mask)

        return logits

    def fit_generator(self, train_data, num_epochs, batch_size, max_num_texts, valid_size=0):
        valid_data = train_data.take(valid_size)
        train_data = train_data.skip(valid_size)

        self.acc_metric.reset_states()
        self.loss_metric.reset_states()

        with trange(num_epochs, desc='Epoch') as pbar:
            for epoch in pbar:
                losses = []
                accs = []

                for i, (Xt_batch, Yt_batch) in enumerate(train_data.cache().shuffle(buffer_size=max_num_texts).batch(batch_size=batch_size)):
                    self.train_step(Xt_batch, Yt_batch)

                    accs.append(self.acc_metric.result().numpy())
                    losses.append(self.loss_metric.result().numpy())

                    np.set_printoptions(precision=4)
                    pbar.set_postfix({
                        'loss': np.nanmean(losses, axis=0),
                        'acc': np.nanmean(accs, axis=0),
                        'batch': '%d/%d'% ((i+1), max_num_texts // batch_size)})
                    pbar.refresh() 
                yield losses

    def fit(self, Xt, Yt, num_epochs, batch_size, max_num_texts):
        train_data = tf.data.Dataset.from_tensor_slices((Xt, Yt))
        return list(self.fit_generator(train_data, num_epochs, batch_size, max_num_texts))

    # def predict(self):
    #     for Xt_batch:
    #         for x:
    #             model(x)

def load_transform_data(max_num_texts, max_num_words, max_vocab_size, src_filename, tar_filename):
    with open(src_filename, 'r') as f:
        X = ['<S> ' + x + ' </S>' for x in tqdm(f, desc='Reading texts')][:max_num_texts]

    with open(tar_filename, 'r') as f:
        Y = ['<S> ' + x + ' </S>' for x in tqdm(f, desc='Reading texts')][:max_num_texts]

    tokX = tf.keras.preprocessing.text.Tokenizer(num_words=max_vocab_size, filters='')
    tokX.fit_on_texts(tqdm(X, desc='Fitting tokenizer'))

    tokY = tf.keras.preprocessing.text.Tokenizer(num_words=max_vocab_size, filters='')
    tokY.fit_on_texts(tqdm(Y, desc='Fitting tokenizer'))

    Xt = tokX.texts_to_sequences(tqdm(X, desc='Transforming texts'))
    Yt = tokY.texts_to_sequences(tqdm(Y, desc='Transforming texts'))

    X_lens = list(map(len, Xt))
    Y_lens = list(map(len, Yt))

    Xt = [x for i, x in enumerate(Xt) if X_lens[i] <= max_num_words and Y_lens[i] <= max_num_words]
    Yt = [y for i, y in enumerate(Yt) if X_lens[i] <= max_num_words and Y_lens[i] <= max_num_words]

    Xt = tf.keras.preprocessing.sequence.pad_sequences(Xt, padding='post', maxlen=max_num_words, truncating='post')
    Yt = tf.keras.preprocessing.sequence.pad_sequences(Yt, padding='post', maxlen=max_num_words, truncating='post')

    print ('vocab_sizes', len(tokX.word_index) + 1, len(tokY.word_index) + 1)
    print ('max_text_lens', max(X_lens), max(Y_lens))

    return Xt, Yt, tokX, tokY


if __name__ == '__main__':
    '''EXAMPLE USAGE'''
    max_num_words = 100
    max_num_texts = 10000
    
    Xt, Yt, tokX, tokY = load_transform_data(max_num_texts=max_num_texts,
                                       max_num_words=max_num_words,
                                       src_filename='data/europarl-v7.de-en.en',
                                       tar_filename='data/europarl-v7.de-en.de')

    print ('num texts', len(Xt), len(Yt))

    model = MyTransformerModule(src_vocab_size=len(tokX.word_index) + 1,
                                tar_vocab_size=len(tokY.word_index) + 1,
                                max_num_words=max_num_words,
                                hidden_size=512,
                                num_enc_layers=3,
                                num_dec_layers=3,
                                num_heads=8,
                                ff_size=2048,
                                keep_prob=1.,
                                learning_rate=0.0001)

    # TODO: plot masks

    print ('trainable variables', len(model.trainable_variables))
    model.fit(Xt, Yt, num_epochs=1000, batch_size=40)