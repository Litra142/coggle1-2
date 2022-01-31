import warnings
warnings.filterwarnings('ignore')
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dropout, Lambda, Bidirectional, Dense, Flatten
from keras.layers.convolutional import Conv1D
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from bulid_input import *
from load_data import *


# 参数设置
BATCH_SIZE = 512
EMBEDDING_DIM = 100
EPOCHS = 20
model_path = 'model/tokenvec_bilstm2_siamese_model.h5'

# 数据准备
train = read_bq('data/bq_corpus/train.tsv',['line_num','q1','q2','label'])

MAX_LENGTH = select_best_length(train)
datas, word_dict = build_data(train)
train_w2v(datas)
VOCAB_SIZE = len(word_dict)
embeddings_dict = load_pretrained_embedding()
embedding_matrix = build_embedding_matrix(word_dict, embeddings_dict,
                                          VOCAB_SIZE, EMBEDDING_DIM)
left_x_train, right_x_train, y_train = convert_data(datas, word_dict, MAX_LENGTH)


def create_bilstm_base_network(input_shape):
    '''搭建Bi-LSTM编码层网络,用于权重共享'''
    input = Input(shape=input_shape)
    lstm1 = Bidirectional(LSTM(128, return_sequences=True))(input)
    lstm1 = Dropout(0.5)(lstm1)

    lstm2 = Bidirectional(LSTM(32))(lstm1)
    lstm2 = Dropout(0.5)(lstm2)

    # 展平
    x_flat = Flatten()(lstm2)
    # 加上三成全连接神经网络
    fc1 = Dense(1024, activation='relu')(x_flat)
    fc2 = Dense(512, activation='relu')(fc1)
    fc3 = Dense(16, activation='relu')(fc2)
    return Model(input, lstm2)


def create_cnn_base_network(input_shape):
    """搭建CNN编码层网络，用于权重共享"""
    input = Input(shape=input_shape)
    cnn1 = Conv1D(64, 3, padding='valid', input_shape=(25, 100), activation='relu', name='conv_1d_layer1')(input)
    cnn1 = Dropout(0.2)(cnn1)
    cnn2 = Conv1D(32, 3, padding='valid', input_shape=(25, 100), activation='relu', name='conv_1d_layer2')(cnn1)
    cnn2 = Dropout(0.2)(cnn2)

    # 展平
    x_flat = Flatten()(cnn2)
    # 加上三成全连接神经网络
    fc1 = Dense(1024, activation='relu')(x_flat)
    fc2 = Dense(512, activation='relu')(fc1)
    fc3 = Dense(16, activation='relu')(fc2)
    return Model(input, cnn2)


def exponent_neg_manhattan_distance(sent_left, sent_right):
    '''基于曼哈顿空间距离计算两个字符串语义空间表示相似度计算'''
    return K.exp(-K.sum(K.abs(sent_left - sent_right), axis=1, keepdims=True))


def bilstm_siamese_model(model_type='lstm'):
    '''搭建网络'''
    # embedding层，其中使用word2vec向量先进行embedding
    embedding_layer = Embedding(VOCAB_SIZE + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_LENGTH,
                                trainable=False,
                                mask_zero=True)
    left_input = Input(shape=(MAX_LENGTH,), dtype='float32', name="left_x")
    right_input = Input(shape=(MAX_LENGTH,), dtype='float32', name='right_x')
    encoded_left = embedding_layer(left_input)
    encoded_right = embedding_layer(right_input)
    if model_type == 'lstm':
        shared_model = create_bilstm_base_network(input_shape=(MAX_LENGTH, EMBEDDING_DIM))
    elif model_type == 'cnn':
        shared_model = create_cnn_base_network(input_shape=(MAX_LENGTH, EMBEDDING_DIM))
    left_output = shared_model(encoded_left)
    right_output = shared_model(encoded_right)
    distance = Lambda(lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                      output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
    model = Model([left_input, right_input], distance)
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
    plt.savefig("model/result_atec.png")
    plt.show()


def train_model(model_type='lstm'):
    '''训练模型'''
    model = bilstm_siamese_model(model_type)
    from keras.utils import plot_model
    plot_model(model, to_file='model/model.png', show_shapes=True)
    history = model.fit(
        x=[left_x_train, right_x_train],
        y=y_train,
        validation_split=0.2,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )
    draw_train(history)
    model.save(model_path)
    return model

train_model(model_type='lstm')
