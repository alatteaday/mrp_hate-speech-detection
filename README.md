# Why Is It Hate Speech? Masked Rationale Prediction for Explainable Hate Speech Detection (COLING 2022)
**Paper link: [COLING](https://aclanthology.org/2022.coling-1.577/)  |  [arXiv](https://arxiv.org/abs/2211.00243)** <br>

## Initial Setting
First, Clone this git repository
```
git clone https://github.com/alatteaday/mrp_hate-speech-detection/
```
The docker commands are implemented in GNU Makefile. You can build the linux-anaconda image that the required packages are installed. The details are written in Dockerfile and Makefile. Or you can use the docker commands directly. 
1. Set the name variables in Makefile, such as IMAGE_NAME, IMAGE_TAG, CONTAINER_NAME, CONTAINER_PORT, NVIDIA_VISIBLE_DEVICES.
2. Build the docker image. Use the Makefile command on the directory the Dockerfile located.
```
make docker-build
```
3. Run the docker container from the built image.
```
make docker-run
```
4. Execute the container.
```
make docker-exec
```

## Models
If you would like to run the models, you can download these compressed files via the Google drive links: ['finetune_1st.tar.gz'](https://drive.google.com/file/d/1BCbgKYNH1-uI_hB18dHRez-Sr3F3VFu4/view?usp=share_link) and ['finetune_2nd.tar.gz'](https://drive.google.com/file/d/1cHpBFWFWq8-o6vLFAbDVcm5Mt2SZY_l8/view?usp=share_link). If you would like to run the final hate speech detection models, you only need the **'finetune_2nd'** and don't have to get 'finetune_1st'. <br>
'finetune_1st' includes the checkpoints of the pre-finetuned models. Each of names shows what's the pre-finetuning method and some infos of the hyperparameters.
```
📁finetune_1st
 ╰─ 📁{checkpoint name}
     ╰─ checkpoint
```
'finetune_2nd' includes the checkpoints of the final models finetuned on hate speech detection. The upper folders indicate which pre-finetuned parameter was used for intialization among the checkpoints in the 'finetune_1st' folder. Each of pre-finetuned checkpoints was finetuned on both two and three-class classification for hate speech detection according to HateXplain benchmark. The two classes are 'non-toxic' and 'toxic', and the three classes are 'normal', 'offensive', and 'hate speech'.

```
📁finetune_2nd
 ╰─ 📁{the pre-finetuned checkpoint name}
     ╰─ 📁{the checkpoint name}
         ╰─ checkpoint 
```

## Test
For testing a model, run second_test.py like below:
```python
python second_test.py -m {the model path to test}
```
If you run a model which trained on two-class detection, it would be tested for Bias-based metrics of hateXplain benchmark. And a model which trained on three-class detection, you could get the results for Performance-based metrics and Explainability-based metrics. 
## Train
To train a hate speech detection model, you need a pre-finetuned model. If you enter the 'bert-based-uncased' instead of the model path as the ```-pf_m``` argument, you can finetune the pre-trained BERT base model on hate speech detection. And you need to set the ```--num_labels``` argument which means the number of the detection class as 2 or 3. 
```python
python second_train.py -pf_m {the pre-finetuned model path to train on hate speech detection} --num_labels {the number of classes: 2 or 3}
```

