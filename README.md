# Dementia Analysis 


## Installation
```
pip install -r requirements.txt
```

## Prepare dataset
First run split dataset to create the train test and dev files.

## Classification
```
python text_classification/run_pl.py
```


## Results
on balanced dataset:
```
{
'model': 'roberta-base',
'acc': 0.9100775193798449,
 'acc_and_f1': 0.9100765466981144,
 'avg_test_loss': 0.2336362898349762,
 'f1': 0.910075574016384,
 'val_loss': 0.2336362898349762
 }
```

unbalanced dataset:
```
{
  'model': "roberta-base",
  'acc': 0.9100775193798449,
  'acc_and_f1': 0.910074925561897,
  'avg_test_loss': 0.2336362898349762,
  'f1': 0.9100723317439489,
  'val_loss': 0.2336362898349762
}
```