import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

class EventDataset(Dataset):
    
    def __init__(self, words, vocab, max_word_length):
    # words all the events
    # vocab list of unique words
    # max_word_length
    
        self.words = words
        self.vocab = vocab
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(vocab)}
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.vocab) + 1 # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_word_length + 1 # <START> token followed by words

    def encode(self, word):
        # adds two characters to token
        # xword = [f'{b}_{a}' for a,b in zip(word,'abcdefgh')]
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        # strips off first 2 characters because ??
        # word = ','.join(self.itos[i][2:] for i in ix)
        word = ','.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        try:
            ix = self.encode(word)
        except:
            print('error __getitem__',idx)
            return None,None
        
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y
    
def create_event_tokens(lines):
    
    pre_tokens = ['<START>']
    embeds = {}
    embeds['<START>'] = (0,0)
     
    for i,line in enumerate(lines):
        for k,w in enumerate(line):
            # pre_tokens.extend([f'{c}_{w}'])
            if w not in pre_tokens:
                pre_tokens.extend([w])
                embeds[w] = [] 
            embeds[w].extend([(i,k)])
            
    # tokens = list(set(pre_tokens))            
                
    return pre_tokens

def create_event_datasets(input_file): 
    """
    each line in event file is 
    0       event,                  # foul,make,miss ... etc
    1,2     period, play_clock,     # 1,12:00 - 0:01
    3,4,5   home_score, away_score, margin, # 100,99,1
    6,7     player, player_team,    # Walter Middy, OKC
    #8,9     home_team, away_team,   # OKC, DAL
    #10      wall_clock              # 7:10 PM
    """
    
    with open(input_file, 'r') as f:
        data = f.read()
    
    # data = data.replace('TIE', '0')
    
    # lines = data.splitlines()[:2000]
    
    lines = data.splitlines() #[:2000]
    max_word_length = max(len(w) for w in lines)
    max_word_length = 16
    # lines = [w.strip() for w in lines] # get rid of any leading or trailing white space
    # strip off last 3 items, home team,away team,wall clock
    lines = [w.split(',') for w in lines if w] # get rid of any empty strings
      
    test_set_size = min(1000, int(len(lines) * 0.1)) # 10% of the training set, or up to 1000 examples
    # rp = torch.randperm(len(lines)).tolist()
    # train_lines = [lines[i] for i in rp[:-test_set_size]]
    # test_lines = [lines[i] for i in rp[-test_set_size:]]
    
    train_lines = lines[:-test_set_size]
    test_lines = lines[-test_set_size:]
    
    print(f"split up the dataset into {len(train_lines)} training examples and {len(test_lines)} test examples")

    vocab = create_event_tokens(lines)
    
    # wrap in dataset objects
    train_dataset = EventDataset(train_lines, vocab, max_word_length)
    test_dataset = EventDataset(test_lines, vocab, max_word_length)

    return train_dataset, test_dataset
