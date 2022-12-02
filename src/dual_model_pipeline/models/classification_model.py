import torch
from transformers import AutoModel, AutoConfig
from sys_config import SysConfig
from collections import OrderedDict

class MedicalClassifier(torch.nn.Module):
    def __init__(self, classifier_name='med_head', config=None) -> None:
        super().__init__()

        sys_config = SysConfig()
        self.n_features = int(sys_config.n_features)
        self.n_labels = int(sys_config.n_labels)

        # Create the transformer base model.
        self.config = config if config else AutoConfig.from_pretrained(sys_config.classification_model_name)
        self.config.output_attentions = True
        self.config.max_position_embedding = sys_config.n_ner_tags
        self.model = AutoModel.from_config(self.config)

        # Create the classification head.
        if classifier_name == 'med_head':
            self.classifier = MediumClassificationHead(self.n_features, self.n_labels, self.config)
        elif classifier_name == 'small_head':
            self.classifier = SmallClassificationHead(self.n_features, self.n_labels, self.config)
        elif classifier_name == 'lrg_head':
            self.classifier = LargeClassificationHead(self.n_features, self.n_labels, self.config)
        else:
            raise ValueError("Parameter 'classifier_name' must be one of: med_head")

    def forward(self, w_x_ids, w_x_attn, t_x_ids, x_features):
        base_output = self.model(input_ids=w_x_ids, attention_mask=w_x_attn, position_ids=t_x_ids)
        x_cls = torch.cat((base_output[0][:,0,:].view(-1, self.config.hidden_size), x_features), dim=1)
        y_hat = self.classifier(x_cls)
        attn = base_output.attentions[-1].mean(dim=1)
        return y_hat, attn

class MediumClassificationHead(torch.nn.Module):
    def __init__(self, n_features, n_labels, config) -> None:
        super().__init__()
        self.config = config
        self.n_features = n_features
        self.n_labels = n_labels
        self.classifier = torch.nn.Sequential(OrderedDict([
            ('linear_1', torch.nn.Linear(self.config.hidden_size + self.n_features, 2048)),
            ('relu_1', torch.nn.ReLU()),
            ('drop_1', torch.nn.Dropout(0.1)),
            ('linear_2', torch.nn.Linear(2048, 1024)),
            ('relu_2', torch.nn.ReLU()),
            ('drop_2', torch.nn.Dropout(0.1)),
            ('classifier', torch.nn.Linear(1024, self.n_labels))
        ]))
    
    def forward(self, x):
        y = self.classifier(x)
        return y

class SmallClassificationHead(torch.nn.Module):
    def __init__(self, n_features, n_labels, config) -> None:
        super().__init__()
        self.config = config
        self.n_features = n_features
        self.n_labels = n_labels
        self.classifier = torch.nn.Sequential(OrderedDict([
            ('linear_1', torch.nn.Linear(self.config.hidden_size + self.n_features, 1024)),
            ('relu_1', torch.nn.ReLU()),
            ('drop_1', torch.nn.Dropout(0.1)),
            ('classifier', torch.nn.Linear(1024, self.n_labels))
        ]))
    
    def forward(self, x):
        y = self.classifier(x)
        return y

class LargeClassificationHead(torch.nn.Module):
    def __init__(self, n_features, n_labels, config) -> None:
        super().__init__()
        self.config = config
        self.n_features = n_features
        self.n_labels = n_labels
        self.classifier = torch.nn.Sequential(OrderedDict([
            ('linear_1', torch.nn.Linear(self.config.hidden_size + self.n_features, 2048)),
            ('relu_1', torch.nn.ReLU()),
            ('drop_1', torch.nn.Dropout(0.1)),
            ('linear_2', torch.nn.Linear(2048, 1024)),
            ('relu_2', torch.nn.ReLU()),
            ('drop_2', torch.nn.Dropout(0.1)),
            ('linear_3', torch.nn.Linear(1024, 512)),
            ('relu_3', torch.nn.ReLU()),
            ('drop_3', torch.nn.Dropout(0.1)),
            ('classifier', torch.nn.Linear(512, self.n_labels))
        ]))
    
    def forward(self, x):
        y = self.classifier(x)
        return y