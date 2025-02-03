# MI201-groupe-4

## Dataset 

Original Kaggle dataset link : 

* [https://www.kaggle.com/competitions/tweet-sentiment-extraction/data](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset/data)

This project is organized into four main folders and one primary notebook directory. Below is an overview of each component and its purpose:

---

### Project Structure

1. **notebooks**  
   - This is the core folder with three notebooks that detail the entire workflow and experiments.
     1. **classification**:  
        - The main notebook where almost all work is consolidated.  
        - Provides answers to all questions and includes experiments centered around the **RoBERTa** model (fine-tuned on a Twitter sentiment classification dataset).  
        - Final results are compared against a larger language model (DeepSeekv3) and the same experiments run on the standard BERT model.  
        - This notebook contains the core of our findings and conclusions.  
     2. **preProcessingAndEDA**:  
        - Provides an initial analysis of the dataset’s statistics without any text preprocessing.  
        - We concluded these features were not very useful based on preliminary tests (not shown in this notebook).  
        - Despite not using them, we decided to include this notebook to document our exploration.  
     3. **BERT**:  
        - Explains the BERT model in detail (answer to question 6) and why we chose it (answer to question 5).  
        - Contains in-depth information, with concise answers summarized at the end.  

(**There is no need to delve into these folders unless you want full transparency about how the calls were made and how the responses were obtained**)
(**All relevant results and procedures were discussed in the classification notebook**)

2. **BERT_BASE_Experiment**  
   - Holds the images and the notebook corresponding to the experiment where we replaced the model used in the main notebook (`classification.ipynb`) with **bert-base-uncased**.  
   - Again, all results (including images) are consolidated and discussed in the main notebook, so this folder serves mainly as a transparent archive of this specific experiment.

3. **Data**  
   - Contains both the original and processed datasets.  
   - Initially, it also stored the generated embeddings; however, due to file size constraints (large files unsuited for GitHub or email), those were removed.  
  
4. **API** 
   - Contains all relevant materials related to API calls made via the llamaAPI to the DeepSeekv3 language model.  
   - This folder includes the Python script, the model’s output, a portion of the log file, and the final generated image.  

---

## Main libraries used
- **Hugging Face Transformers** : Embedding generation (BERT/roBERTa).
- **scikit-learn** : For model training, selection, evaluation, pipelining...
- **matplotlib/seaborn** : Result visualization
- **Pytorch**: Neural networks
- **Pandas/Numpy** : Data manipulation
