import warnings
warnings.filterwarnings('ignore')
from keras.layers import *
from keras.activations import softmax
from keras.models import Model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from bulid_input import *
from load_data import *
import matplotlib.pyplot as plt

def StaticEmbedding(embedding_matrix):
    # Embedding metrix
    in_dim, out_dim = embedding_matrix.shape
    return Embedding(in_dim, out_dim, weights=[embedding_matrix], trainable=False)


def subtract(input_1, input_2):
    minus_input_2 = Lambda(lambda x: -x)(input_2)
    return add([input_1, minus_input_2])


def aggregate(input_1, input_2, num_dense=300, dropout_rate=0.5):
    feat1 = concatenate([GlobalAvgPool1D()(input_1), GlobalMaxPool1D()(input_1)])
    feat2 = concatenate([GlobalAvgPool1D()(input_2), GlobalMaxPool1D()(input_2)])
    x = concatenate([feat1, feat2])
    x = BatchNormalization()(x)
    x = Dense(num_dense, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(num_dense, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return x


def align(input_1, input_2):
    attention = Dot(axes=-1, name='attention-layer')([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def build_model(embedding_matrix, num_class=1, max_length=30, lstm_dim=300):
    q1 = Input(shape=(max_length,))
    q2 = Input(shape=(max_length,))

    # Embedding
    embedding = StaticEmbedding(embedding_matrix)
    q1_embed = BatchNormalization(axis=2)(embedding(q1))
    q2_embed = BatchNormalization(axis=2)(embedding(q2))

    # Encoding
    encode = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)

    # Alignment
    q1_aligned, q2_aligned = align(q1_encoded, q2_encoded)

    # Compare
    q1_combined = concatenate(
        [q1_encoded, q2_aligned, subtract(q1_encoded, q2_aligned), multiply([q1_encoded, q2_aligned])])
    q2_combined = concatenate(
        [q2_encoded, q1_aligned, subtract(q2_encoded, q1_aligned), multiply([q2_encoded, q1_aligned])])
    compare = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q1_compare = compare(q1_combined)
    q2_compare = compare(q2_combined)

    # Aggregate
    x = aggregate(q1_compare, q2_compare)
    x = Dense(num_class, activation='sigmoid')(x)
    model = Model(inputs=[q1, q2], outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
    model.summary()
    return model

def draw_train(history):
    '''绘制训练曲线'''
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("model/result_esim.png")
    plt.show()


if __name__ == "__main__":
    # 参数设置
    BATCH_SIZE = 512
    EMBEDDING_DIM = 100
    EPOCHS = 20
    model_path = 'model/tokenvec_esim_model.h5'

    # 数据准备
    train = read_bq('data/bq_corpus/train.tsv', ['line_num', 'q1', 'q2', 'label'])

    MAX_LENGTH = select_best_length(train)
    datas, word_dict = build_data(train)
    train_w2v(datas)
    VOCAB_SIZE = len(word_dict)
    embeddings_dict = load_pretrained_embedding()
    embedding_matrix = build_embedding_matrix(word_dict, embeddings_dict,
                                              VOCAB_SIZE, EMBEDDING_DIM)
    left_x_train, right_x_train, y_train = convert_data(datas, word_dict, MAX_LENGTH)
    model = build_model(embedding_matrix, max_length=MAX_LENGTH, lstm_dim=128)
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='model/model_esim.png', show_shapes=True)
    history = model.fit(
        x=[left_x_train, right_x_train],
        y=y_train,
        validation_split=0.2,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )
    draw_train(history)
    model.save(model_path)
