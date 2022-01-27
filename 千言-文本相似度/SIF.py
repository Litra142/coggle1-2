from gensim.models import Word2Vec


class SIF:
    def __init__(self, a, dim):
        self.a = a          # the parameter in the SIF weighting scheme, usually in 
        self.dim = dim      #词向量的维度
        self.model_path = '.\word2vec_1.model'
        # self.model2_path = '.\word2vec_2.model'
        self.weight_file = '' # 这个我好像没有保存下来，待会去保存一下
        self.sen_file = 'sentences1.txt'
        self.sent = []     # 句子列表
        # self.sen2_file = 'sentences2.txt'

    def load_w2v(self):
        self.w2v_model = Word2Vec.load(self.model_path)
        return self.w2v_model
    
    def read_sentences(self):
        with open(self.sen_file, encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                if line:
                    line = line.strip()
                    self.sent.append(line.split())
        # length = len(self.sent)       # 句子列表长度
        return self.sent
    
    def save_dict(self):
        pass
