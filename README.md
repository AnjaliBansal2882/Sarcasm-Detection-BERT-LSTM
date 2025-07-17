# <ins>Sarcasm Detection in News<ins>

#### I have finally gathered time to post about my previous work. Starting from this repository, which is the implementation of one of my in-process research paper on Sarcasm Detection using context based LLM models like BERT and LSTM. The ptbn are as follows:

- The models were fine tuned on the News Headlined Dataset for Sracasm Detection. the dataset has 30k sarcastic news headlines. The data in this dataset has been gathered from _The Onion_ and _The HuffPost_.

- The dataset has also been further divided into short and long, depending on the number of words in the headlines, which alters the accuracy of the models.

- 2 models namely **LSTM** (Long Short Term Memory) and **BERT** (Bi-Directional Encoder Representation from Transformers) have been compared. 

- I also performed various optimizations on each of the models for better validation loss and precision values. 

- The optimizations performed on **_LSTM_** model are as follows:
    
    - L2 Regularization to reduce over fitting
    - Batch Normalization and L2 regularization for faster training

- The optimizations performed on the **_BERT_** are as follows:

    - Layer Freezing

Accuracy and time metrics are for the above mentioned model configurations are as follows:
------------------------
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; text-align: center;">
  <tr>
    <th>Model</th>
    <th>Accuracy</th>
    <th>Training Time</th>
  </tr>
  <tr>
    <td>Default LSTM</td>
    <td>85.91</td>
    <td>370</td>
  </tr>
  <tr>
    <td>LSTM + L2</td>
    <td>86.28</td>
    <td>439</td>
  </tr>
  <tr>
    <td>LSTM + BN + L2</td>
    <td>87.32</td>
    <td>314</td>
  </tr>
  <tr>
    <td>Default BERT</td>
    <td>93.11</td>
    <td>3621</td>
  </tr>
  <tr>
    <td>BERT + Layer Freeze</td>
    <td>90.63</td>
    <td>3639</td>
  </tr>
</table>
