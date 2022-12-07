from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import torch
import nltk.data
from dual_model_pipeline.data.ner_medmentions_load import get_ner_tags
from sys_config import SysConfig
import os

nltk.download('punkt')

class NER_Model():

    def __init__(self, device=None) -> None:
        self.tags2id, self.id2tags = get_ner_tags() 
        config = SysConfig()
        self.sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        self.model_name = config.ner_model_name
        self.checkpoint = os.path.join(config.ner_model_path, config.ner_checkpoint)
        self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.checkpoint)
        self.pipeline = pipeline(task='token-classification', model=self.model, tokenizer=self.tokenizer, device=self.device)

    def inference(self, utterance):
        sentences = self.sent_detector.tokenize(utterance.strip())
        words = []
        tags = []
        confidence = []
        indices = []

        i = 0
        for sentence in sentences:
            out = self.pipeline(sentence)
            for token in out:
                entity = self.id2tags[int(token['entity'][6:])]
                if entity == 'O':
                    continue
                
                word = token['word']

                if word.startswith("##"):
                    if len(words) == 0:
                        continue
                    words[-1] += word[2:]
                elif entity.startswith("I-") and len(tags) > 0 and entity[2:] == tags[-1]:
                    words[-1] += " "+word
                else:
                    words.append(word)
                    tags.append(entity[2:])
                    confidence.append(float(token['score']))
                    indices.append(i)

                i += 1

        return words, tags, confidence, indices