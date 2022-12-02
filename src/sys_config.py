import configparser
import socket
import os

class SysConfig():
    def __init__(self) -> None:
        self.hostname = socket.gethostname()
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join('system.ini'))

    @property
    def ner_model_name(self):
        return self.config[self.hostname]['ner_model_name']
    
    @property
    def ner_checkpoint(self):
        return self.config[self.hostname]['ner_checkpoint']

    @property
    def classification_model_name(self):
        return self.config[self.hostname]['classification_model_name']

    @property
    def classification_model_checkpoint(self):
        return self.config[self.hostname]['classification_model_checkpoint']

    @property
    def classification_model_size(self):
        return self.config[self.hostname]['classification_model_size']

    @property
    def classification_model_path(self):
        return self.config[self.hostname]['classification_model_path']
    
    @property
    def n_features(self):
        return self.config[self.hostname]['n_features']

    @property
    def n_labels(self):
        return self.config[self.hostname]['n_labels']

    @property
    def n_ner_tags(self):
        return self.config[self.hostname]['n_ner_tags']