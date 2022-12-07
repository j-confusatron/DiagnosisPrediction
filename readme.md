# Diagnosis Prediction System
This system combines NLP utterances in the form of physician notes with patient contextual data to produce a diagnosis classification. It does so by using a two-model pipeline.

The first model is a NER classifier. This model applies a UMLS-21 codes to words, thus allowing the pipeline to identify interesting medical terminology in the utterance. All non-O tags are aligned with their words. The tags and words are each used to create their own embeddings.

The tag and word embeddings are added to each other and this forms the input to the second model, the diagnosis classifier. This model creates a NLP latent state representation of the tags and words. This state, in vector form, is concatenated with patient contextual data to form the input to a classification head that produces the diagnosis classification.

All of this is done in a way that is highly interpretable. The provided server and client allow users of the system to see not only the most likely diagnosis, but all diagnoses, how confident the system was in those classifications, which words were identified as UMLS terms, and how much attention was paid to each of those UMLS terms when the diagnosis was predicted.

## Acknowledgements
Both the NER and classification models use KrissBert (https://huggingface.co/microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL)  as their underlying language model. KrissBert is a transformer that uses the Bert architecture, trained specifically on medical texts.

The MedMentions (https://github.com/chanzuckerberg/MedMentions) dataset is used to train the NER instance of KrissBert.

The IMR (https://catalog.data.gov/dataset/independent-medical-review-imr-determinations-trend) dataset us used to train the diagnosis classifier.

## System Setup
1. Open a CLI at project root
2. Create a Python virtual environment: `conda create --name <env> --file requirements.txt`
3. Activate the virtual environment
4. Navigate to `/src`
5. Edit system.ini paths (see below)
6. It is highly recommended to have CUDA support before training models
7. Train the NER UMLS model: `python -m dual_model_pipeline.ner_trainer`
8. Train the Diagnosis Classification model: `python -m dual_model_pipeline.classification_trainer`
9. Start the server: `python -m server`
10. Interact with the server using the pre-built client at: `\client\client.html`

### system.ini
All options under the `[DEFAULT]` header can be left alone.

You'll need to know your machines name. From a CLI with your virtual environment (above) active:
1. `python` to start a Python 3.8 interpreter
2. `import socket`
3. `print(socket.gethostname())` to get your machine's name

Use your machine's name to create a new sub-section in system.ini. It should look like:

`[Dantooine]`

`ner_model_path = d:\medlangmodel\ner\`

`classification_model_path = d:\medlangmodel\classification\`

`ner_model_path` and `classification_model_path` are the paths where the trainers will write models and where the server will read models from.