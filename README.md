# SMS Spam Detection Using LSTM

A Python deep learning project that classifies SMS messages as **Spam** or **Ham (legitimate)** using an LSTM neural network built with PyTorch.

## Features

- Automatic dataset download
- Text preprocessing and tokenization
- LSTM-based text classification
- Evaluation using accuracy, precision, recall, F1-score, and a confusion matrix
- Interactive terminal predictions for custom SMS messages

## Tech Stack

- Python
- PyTorch
- scikit-learn
- LSTM (Long Short-Term Memory)

## Dataset

This project uses the [UCI SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), containing 5,574 labeled SMS messages.

## Project Structure

```text
SMS-spam-LSTM/
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

The dataset and trained model are generated locally and excluded from Git.

## Setup

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Train the Model

```powershell
python train.py
```

This downloads the dataset if needed, trains the model, evaluates its performance, and saves the trained checkpoint in `results/spam_lstm.pt`.

## Predict SMS Messages

```powershell
python predict.py
```

Enter a message to see the predicted class and estimated spam probability. Type `exit` to quit.

## Evaluation Results

On a stratified test split of 1,115 messages:

| Metric | Score |
|---|---:|
| Accuracy | 98.65% |
| Spam precision | 99.26% |
| Spam recall | 90.60% |
| Spam F1-score | 94.74% |

These results are from one test split. Performance on unseen messages may differ.

## Future Improvements

- Compare LSTM with traditional machine learning models
- Tune the prediction threshold
- Test on additional unseen SMS messages
- Improve text preprocessing

## License

Educational project. Refer to the dataset provider's terms for dataset usage and redistribution.

