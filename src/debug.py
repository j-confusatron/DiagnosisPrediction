import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

labels = [
    "Digestive System/ GI",
    "Skin Disorders",
    "CNS/ Neuromusc Dis",
    "Cancer",
    "Pediatrics",
    "Cardiac/Circ Problem",
    "Foot Disorder",
    "Chron Pain Synd",
    "Genetic Diseases",
    "Vision",
    "Immuno Disorders",
    "Not Applicable",
    "Prevention/Good Hlth",
    "Dental Problems",
    "Infectious Disease",
    "GU/ Kidney Disorder",
    "Morbid Obesity",
    "Orth/Musculoskeletal",
    "Respiratory System",
    "Trauma/ Injuries",
    "Pregnancy/Childbirth",
    "Mental Disorder",
    "Blood Related Disord",
    "Endocrine/Metabolic",
    "OB-GYN/ Pregnancy",
    "Autism Spectrum",
    "Other",
    "Post Surgical Comp",
    "Ears/Nose/Throat"
]


df_cm = pd.DataFrame(np.random.rand(len(labels),len(labels)), index = labels,
                  columns = labels)
plt.figure(figsize = (12,10))
sn.set(font_scale=.9)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 6})
plt.ylabel("Ground Truth")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()