import evaluate, json, numpy, torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, BertTokenizerFast, AlbertTokenizerFast, AutoModelForSequenceClassification
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, BertForSequenceClassification, AlbertForSequenceClassification
from transformers import GPT2ForSequenceClassification, XLNetForSequenceClassification, XLNetTokenizer
from torch.nn import functional as F

from transformers import set_seed

from transformers import GPT2Tokenizer
import re

set_seed(2023)

text_form="em_mc"
text_form="em"


device = "cuda:0" if torch.cuda.is_available() else "cpu"
#print("device is:",device)

#tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#tokenizer.pad_token = tokenizer.eos_token

#tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = numpy.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def save_jsonline(data,savepath):
    print("data is",data)
    json_object = json.dumps(str(data), indent=4)
    with open(savepath,'w') as f:
        f.write(json_object)    
    return 

def load_data(q_path="test/1000_perf_questions.json",
        a_path="test/1000_perf_answers.json",
        label_path='test/1000_perf_em_list.txt',):
  print("label path",label_path)
  q1 = json.load(open(q_path))
  a1 = json.load(open(a_path))
  labels = numpy.loadtxt(label_path,dtype=int)
  text = list()
  for item in q1:
    id1 = item['_id']
    q_cur = item['query'].split("\n\n")[-1]
    ans1 = a1['answer'][id1]
    if(text_form=="em_mc"):
        ans1=ans2opt(ans1)
    query = q_cur+" "+ans1
    text.append(query)
  return text, labels
  
def ans2opt(ans1):
    def mc_remove(text):
        a1 = re.findall('\([a-zA-Z]\)', text)
        #print("text is",text)
        #print("a1",a1)
        if(len(a1)==0):
            return ""
        return re.findall('\([a-zA-Z]\)', text)[-1]
    ans2 = mc_remove(ans1)
    return ans2  

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class Score(object):
    def __init__(self,          
                 score_type='DistilBert'):
        if(score_type=='DistilBert'):
            self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        if(score_type=='Bert'):
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        if(score_type=='AlBert'):
            self.tokenizer = AlbertTokenizerFast.from_pretrained('albert-base-v2')
        self.score_type = score_type
        return
    def train(self,
              train_texts, 
              train_labels,
              #score_type='DistilBert',
              ):
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.6)
        #print("train_text 0",train_texts[0])
        #print("val_text 0",val_texts[0])
        #print("----------------------------")
        tokenizer = self.tokenizer
        train_encodings = tokenizer(train_texts, truncation=True, padding=True,max_length=512)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True,max_length=512)

        train_dataset = IMDbDataset(train_encodings, train_labels)
        val_dataset = IMDbDataset(val_encodings, val_labels)

        training_args = TrainingArguments(
    output_dir='./scorer_location',          # output directory
    num_train_epochs=8,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",
	save_strategy ="epoch",
    load_best_model_at_end=True,
    seed=2023,
    )
        score_type = self.score_type
        if(score_type=='DistilBert'):
            model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        if(score_type=='Bert'):
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
        if(score_type=='AlBert'):
            model = AlbertForSequenceClassification.from_pretrained("albert-base-v2")

        #model = GPT2ForSequenceClassification.from_pretrained("gpt2")
        #model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased")

        model = model.to(device)
        trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset ,            # evaluation dataset
    compute_metrics=compute_metrics,

        )

        trainer.train()
        self.trainer = trainer
        self.model = model
        return model
    
    def predict(self,
                model,text):
        #trainer = self.trainer
        model = self.model
        encoding = self.tokenizer(text, return_tensors="pt",truncation=True, padding=True)
        encoding = {k: v.to(model.device) for k,v in encoding.items()}
        outputs = model(**encoding)
        logit_score = outputs.logits.cpu().detach()
        #return logit_score
        # convert logit score to torch array
        torch_logits = logit_score
        
        # get probabilities using softmax from logit score and convert it to numpy array
        probabilities_scores = F.softmax(torch_logits, dim = -1).numpy()[0]

        return probabilities_scores
    
    def get_model(self):
        return self.model

    def get_score(self,text):
        prob = self.predict("",text)
        return prob[1]

    def gen_score(self,
                  model,
                  texts,
                  ):
        scores = list()
        for text in texts:
            prob = self.predict(model,text)
            scores.append(prob[1])
        return scores
    def save(self,savepath):
        self.trainer.save_model(savepath)
        return

    def load(self,loadpath):
        self.model = AutoModelForSequenceClassification.from_pretrained(loadpath)
        return
    def save_scores(self,
                    q_path,
                    score_path,
                    scores):
        q1 = json.load(open(q_path))
        result = dict()
        i = 0
        for item in q1:
            id1 = item['_id']
            result[id1] = scores[i]
            i+=1
        save_jsonline(data=result, savepath=score_path)    
        return
    def pipelines(self,
                  train_q_path,
                  train_a_path,
                  train_label_path,
                  train_score_path,
                  
                  val_q_path,
                  val_a_path,
                  val_label_path,
                  val_score_path,
                  
                  test_q_path,
                  test_a_path,
                  test_label_path,
                  test_score_path,
                  ):
        # generate data
        train_texts, train_labels = load_data(
                q_path=train_q_path,
                a_path=train_a_path,
                label_path=train_label_path)

        test_texts, test_labels = load_data(
                q_path=test_q_path,
                a_path=test_a_path,
                label_path=test_label_path)

        val_texts, val_labels = load_data(
                q_path=val_q_path,
                a_path=val_a_path,
                label_path=val_label_path)
        
        # train the model
        model = self.train(train_texts, train_labels)
        
        # get scores 
        scores = self.gen_score(model,test_texts)
        self.save_scores(test_q_path,test_score_path,scores)
        scores = self.gen_score(model,val_texts)
        self.save_scores(val_q_path,val_score_path,scores)
        scores = self.gen_score(model,train_texts)
        self.save_scores(train_q_path,train_score_path,scores)
        return
    
def main():
    print("test of scoring functions")
    MyScore = Score()
    train_q_path="../../api_performance/headlines/9_train/1000_perf_questions.json"
    train_a_path="../../api_performance/headlines/9_train/1000_perf_answers.json"
    train_label_path="../../api_performance/headlines/9_train/1000_perf_em_list.txt"
    train_score_path="../../api_performance/headlines/9_train/1000_perf_scores.json"

    val_q_path="../../api_performance/headlines/9_train/1000_perf_questions.json"
    val_a_path="../../api_performance/headlines/9_train/1000_perf_answers.json"
    val_label_path="../../api_performance/headlines/9_train/1000_perf_em_list.txt"
    val_score_path="../../api_performance/headlines/9_train/1000_perf_scores.json"
                  
    test_q_path="../../api_performance/headlines/9/1000_perf_questions.json"
    test_a_path="../../api_performance/headlines/9/1000_perf_answers.json"
    test_label_path="../../api_performance/headlines/9/1000_perf_em_list.txt"
    test_score_path="../../api_performance/headlines/9/1000_perf_scores.json"
                  
    MyScore.pipelines(train_q_path, train_a_path, train_label_path, train_score_path, val_q_path, val_a_path, val_label_path, val_score_path, test_q_path, test_a_path, test_label_path, test_score_path)
    return    
if __name__ == "__main__":
   main()