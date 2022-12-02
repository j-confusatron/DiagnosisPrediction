import torch
from transformers import AutoTokenizer
import numpy as np
import os

from dual_model_pipeline.models.ner_model import NER_Model
from dual_model_pipeline.models.classification_model import MedicalClassifier
from dual_model_pipeline.data.classification_imr_load import get_mappings
from dual_model_pipeline.data.ner_medmentions_load import get_ner_tags
from sys_config import SysConfig
from util import one_hot

class DiagnosisPredicter():

    def __init__(self) -> None:
        self.config = SysConfig()
        self.device = torch.device('cpu')

        self.ner_tagger = NER_Model(self.device)
        self.ner2idx, self.idx2ner = get_ner_tags(root_only=True)
        
        cls_model = MedicalClassifier(self.config.classification_model_size)
        f_model = os.path.join(self.config.classification_model_path, self.config.classification_model_checkpoint, 'model.pt')
        cls_model.load_state_dict(torch.load(f_model, map_location=self.device))
        self.cls_model = cls_model.to(self.device)
        self.cls_model.eval()
        self.cls_tokenizer = AutoTokenizer.from_pretrained(self.config.classification_model_name)
        self.cls_mappings = get_mappings()

    def inference(self, data):
        # Get the NER terms.
        ner_words, ner_tags, ner_confidence, ner_indices = self.ner_tagger.inference(data['utterance'])
        ner = []
        for word, tag, score, idx in zip(ner_words, ner_tags, ner_confidence, ner_indices):
            ner.append({'word': word, 'tag': tag, 'score': score, 'idx': idx})

        # Collect the patient context.
        idx_age = self.cls_mappings['AgeRange']['AgeRange2idx'][data['age']]
        idx_gender = self.cls_mappings['PatientGender']['PatientGender2idx'][data['gender']]
        age = one_hot(idx_age, self.cls_mappings['AgeRange']['AgeRange_len'])
        gender = one_hot(idx_gender, self.cls_mappings['PatientGender']['PatientGender_len'])
        x_context = torch.tensor(age+gender).unsqueeze(0).to(self.device)

        # Tokenize the NER terms.
        word_toks = self.cls_tokenizer(ner_words, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
        tag_toks = np.zeros(len(word_toks['offset_mapping']), dtype=np.int32)
        tags = ['O']+ner_tags
        i_tag = -1
        for i, w_id in enumerate(word_toks['offset_mapping']):
            if w_id[0] == 0:
                i_tag += 1
                if i_tag >= len(tags):
                    break
            tag_toks[i] = self.ner2idx[tags[i_tag]]
        x_words = torch.tensor(word_toks['input_ids']).unsqueeze(0).to(self.device)
        x_attn = torch.tensor(word_toks['attention_mask']).unsqueeze(0).to(self.device)
        x_tags = torch.tensor(tag_toks).unsqueeze(0).to(self.device)

        # Predict diagnosis.
        y_hat, attn = self.cls_model(x_words, x_attn, x_tags, x_context)
        y_hat = torch.softmax(y_hat.view(-1), -1).detach().cpu().numpy()
        attn = attn.view(-1, len(tag_toks)).detach().cpu().numpy()[0]
        labels = [
            {
                'idx': i, 
                'label': self.cls_mappings['DiagnosisCategory']['idx2DiagnosisCategory'][i], 
                'score': float(score)
            } for i, score in enumerate(y_hat)
        ]
        diagnosis = {
            'diagnosis': self.cls_mappings['DiagnosisCategory']['idx2DiagnosisCategory'][np.argmax(y_hat)],
            'score': float(np.max(y_hat))
        }

        # Align attention with logical tokens.
        attn_words = ['CLS']+ner_words+['PAD']
        logical_attn = np.zeros(len(attn_words))
        i_attn = -1
        for w_idx, w_attn in zip(word_toks['offset_mapping'], attn):
            if w_idx[0] == 0:
                i_attn += 1
                if i_attn >= len(attn):
                    break
            logical_attn[i_attn] += w_attn
        attn = [{'word': w, 'attn': float(a)} for w, a in zip(attn_words, logical_attn)]

        # Return the findings.
        result = {'diagnosis': diagnosis, 'ner': ner, 'labels': labels, 'attention': attn}
        return result