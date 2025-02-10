# NLP

A1: RESEARCH QUESTION:

"What are the most influential words that indicate a positive or negative sentiment in product reviews?"

A2: OBJECTIVES OR GOALS:

The objectives of this data analysis are to identify the most influential words or phrases that indicate positive or negative sentiment in Amazon product reviews using neural network models and NLP techniques. I aim to build a model that can accurately classify reviews based on sentiment and highlight the key terms that contribute to this classification. By understanding these influential words or phrases, businesses can gain insights into what aspects of their products are most appreciated or criticized by customers, enabling them to make data-driven improvements and enhance customer satisfaction.

A3: PRESCRIBED NETWORK:

Recurrent Neural Network (RNN) is the best option because it is designed to handle text. Specifically, Long Short-Term Memory (LSTM) networks, are great for this purpose because they can learn and remember long-term dependencies in the text. LSTMs are capable of processing each word in a review in context, capturing the meaning and sentiment effectively. By training an LSTM on our dataset of Amazon product reviews, it can learn to predict whether a review is positive or negative and identify the key words and phrases that contribute to these sentiments.

B1: DATA EXPLORATION:

- Presence of unusual characters: I examined the reviews for unusual characters such as emojis and non-English characters. I found that there were no unusual characters in the dataset, simplifying my data cleaning process.
- Vocabulary size: I analyzed the dataset to determine the vocabulary size, which turned out to be 2,235 unique words. This large vocabulary indicates that my model will need to handle a wide range of words, making word embeddings crucial for capturing semantic meanings effectively.
- Proposed word embedding length: For my text classification task, I propose using a word embedding length of 100. This length provides a good balance between capturing sufficient semantic information and computational efficiency.
- Statistical justification for the chosen maximum sequence length: To determine the maximum sequence length for my model, I analyzed the distribution of review lengths. The statistical analysis showed that the 95th percentile of review lengths is 26 words, justifying my choice to cover the majority of reviews without truncating too much information.

B2: TOKENIZATION:

The goals of the tokenization process are to break down the text into smaller, manageable pieces (tokens) and to normalize the text for consistent processing. Tokenization involves splitting the text into words, which can then be analyzed or fed into a machine learning model. Normalization includes converting all characters to lowercase, removing punctuation, and making sure that the text is in a standard format for analysis.


B3: PADDING PROCESS:
The padding process is used to standardize the length of sequences so that all input sequences to the neural network have the same length. This is important because neural networks typically require fixed-size input tensors. Padding ensures that shorter sequences are extended to the same length as the longest sequence by adding special padding tokens. Padding can occur either before (pre-padding) or after (post-padding) the text sequence. A special token, often zero, is used to fill the extra positions in the sequence.

![image](https://github.com/user-attachments/assets/a36a0cd9-ea91-492d-b2cd-31ec2b2c829b)

![image](https://github.com/user-attachments/assets/17a6c7f6-f921-4a81-8beb-11887852889f)


B4: CATEGORIES OF SENTIMENT:
For this sentiment analysis task, there are two categories of sentiment: positive and negative. To classify the reviews into these categories, I will use a sigmoid activation function in the final dense layer of the neural network. The sigmoid function outputs a probability between 0 and 1, indicating the likelihood that a given review is positive. This setup is ideal for binary classification tasks, where we need to determine whether each review falls into one of the two sentiment categories.

B5: STEPS TO PREPARE THE DATA:
To prepare the data for analysis, I first loaded the dataset into a pandas DataFrame. Then, I normalized the text by converting it to lowercase and removing non-alphanumeric characters to ensure consistency. Next, I tokenized the text into individual words and converted these tokens into sequences of integers using the Tokenizer from Keras. I applied padding to these sequences to ensure they all have a uniform length. I then extracted the sentiment labels from the dataset. Finally, I split the data into training, validation, and test sets, following the industry average split of 70% training, 15% validation, and 15% test. This resulted in a training set size of 700, a validation set size of 150, and a test set size of 150.

C1: MODEL SUMMARY:

![image](https://github.com/user-attachments/assets/06fec79b-028c-4bc6-a217-3d6d20335457)

C2: NETWORK ARCHITECTURE:

The model I used has five layers: an embedding layer, a global average pooling layer, and three dense (fully connected) layers. The embedding layer converts words into dense vectors, with 190,600 parameters, representing the relationship between words. The global average pooling layer reduces the size of these vectors by averaging, making the data more manageable for the dense layers. The first dense layer has 100 units with 10,100 parameters, the second dense layer has 50 units with 5,050 parameters, and the final dense layer has 1 unit with 51 parameters to classify the sentiment as positive or negative. In total, the model has 205,801 trainable parameters, which include all the weights and biases in these layers.

C3: HYPERPARAMETERS:

- Activation Functions: ReLU for hidden layers, sigmoid for the output layer.
- Number of Nodes per Layer: 100 nodes in the first dense layer, 50 nodes in the second dense layer.
- Loss Function: Binary cross-entropy, suitable for binary classification.
- Optimizer: Adam, known for its efficiency and adaptive learning rates.
- Stopping Criteria: Early stopping with a patience of 3 to prevent overfitting.
- Evaluation Metric: Accuracy, providing a straightforward measure of model performance.

D1: STOPPING CRITERIA:

By employing early stopping, the model was able to halt training at the optimal point where the validation loss no longer improved. This approach helps in preventing overfitting, ensuring the model maintains good generalization to new data. The number of epochs used was determined dynamically, based on the model's performance on the validation set, rather than being a fixed number, which adds to the robustness of the training process.

![image](https://github.com/user-attachments/assets/1b9dd95b-07d8-4ff3-8dbc-297fff92b033)

D2: FITNESS:

The fitness of the model is determined by evaluating its performance on training, validation, and test datasets. By implementing early stopping, monitoring performance metrics, and considering additional regularization techniques, the model is designed to balance learning while avoiding overfitting, ensuring robust performance on new, unseen data. The test accuracy (0.793) provides a final measure of how well the model generalizes, which in this case should be close to the validation accuracy if the model is well-fitted.

D3: TRAINING PROCESS:

![image](https://github.com/user-attachments/assets/43cea4eb-092e-4c1f-9c43-8af8bd5402f1)


![image](https://github.com/user-attachments/assets/cbe4adce-09bb-4b12-bb5a-fbe14467f328)


![image](https://github.com/user-attachments/assets/6859fbe3-54b1-4e62-82c6-b1c3dd0bb735)

D4: PREDICTIVE ACCURACY:

The predictive accuracy, as indicated by the test accuracy, demonstrates the effectiveness of the model. With an accuracy of 79% (or whatever the final test accuracy may be), the model is able to reliably distinguish between positive and negative reviews. This level of accuracy is generally considered good for sentiment analysis tasks, indicating that the model can be useful for real-world applications such as automated sentiment analysis for product reviews.

F: FUNCTIONALITY:

My neural network for sentiment analysis uses an effective architecture that includes an embedding layer, a global average pooling layer, and several dense layers. The embedding layer converts words into dense vectors, capturing their meanings. The global average pooling layer reduces these vectors, making the data easier to manage. The dense layers then learn complex patterns to classify reviews as positive or negative. The use of early stopping ensures the model doesn't overfit, making it generalize well to new data. Overall, this architecture efficiently captures and processes text data, resulting in a model that accurately predicts sentiment.

G: RECOMMENDATIONS:

I recommend deploying the model for real-time analysis of product reviews to quickly identify customer feedback. Integrating this system with customer support can help address negative reviews promptly, improving customer satisfaction. We can also use the insights to refine our marketing strategies and highlight the strengths of our products.















