PAD_token = 0 # for padding short sentences
SOS_token = 1 # Start of sentence
EOS_token = 2 # end of stream

class Vocab:
    """"
    Vocab 
    """
    def __init__(self, name='vocab', trim_count=3) -> None:
        self.name = name
        self.trim_count = trim_count
        self.trimmed = False
        tokens = []

        self.word2int = {'PAD':PAD_token, 'SOS':SOS_token, 'EOS':EOS_token}
        self.int2word = {PAD_token:'PAD', SOS_token:'SOS', EOS_token:'EOS'}
        self.word2count = {}
        self.numb_words = 3 # tokens added 

    # add word to dicts
    def add_word(self, word):
        if word not in self.word2int:
            self.word2int[word] = self.numb_words
            self.int2word[self.numb_words] = word
            self.word2count[word] = 1
            self.numb_words += 1
        else:
            self.word2count[word] +=1
        
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word=word)

    def trim(self):
        """trim vocab with frequency less than self.trim_count"""
        if self.trimmed:
            print('Vocab already trimmed')
            return
        self.trimmed = True

        keep_words = []
        keep_count = self.word2count
        for word, count in self.word2count.items():
            if count >= self.trim_count:
                keep_words.append(word)

        print('........This is print from Voc class after call to trimm method.........')
        print('keep_words {} / {} = {:.4f}'.format(len(keep_words), 
              len(self.word2int), len(keep_words) / len(self.word2int)))
        print('..........................End of print..................................\n')

        #reinitialize dicts  
        self.word2int = {'PAD':PAD_token, 'SOS':SOS_token, 'EOS':EOS_token}
        self.int2word = {PAD_token:'PAD', SOS_token:'SOS', EOS_token:'EOS'}
        self.word2count = {}
        self.numb_words = 3 # tokens added 
        for word in keep_words:
            self.add_word(word)

        #keep original word_count value for untrimmed words
        for word in self.word2count:
            if word in keep_count:
                self.word2count[word] = keep_count[word]