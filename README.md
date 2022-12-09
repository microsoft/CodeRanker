# Fault-aware Neural Code Rankers

> This repo has the code accompanying the CodeRanker NeurIPS'22 [paper](https://arxiv.org/pdf/2206.03865.pdf). 

## Installation
```
pip install -r requirements.txt
```

## Dataset 
The ranker datasets are available through GIT LFS in dataset/ directory


## Usage
First, cd into src directory

1. Training rankers
```
bash finetune.sh DATADIR MODELDIR CACHEDIR TASK
```
where DATADIR is the location of the directory containing the desired ranker dataset  
MODELDIR is the location of the output dirctory where the trained model will be stored  
CACHEDIR is the location of the cache directory both for caching the model and for caching the dataset  
TASK is one of binary, ternary, intent_error, execution_error, execution_error_with_line

2. Inference with rankers
```
bash eval.sh DATADIR MODELDIR CACHEDIR TASK {val|test}.json PREDICT_FILENAME
```
where PREDICT_FILENAME is the name of the file inside MODELDIR where the inferenced logits will be stored. 

3. Computing the ranked metrics
```
python3 compute_metrics.py --data_file=DATADIR/{val|test}.json --logits_prediction_file=MODELDIR/PREDICT_FILENAME --labels_file=DATADIR/labels_TASK.txt --task=TASK
```


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
