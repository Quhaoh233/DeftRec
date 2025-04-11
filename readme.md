# DeftRec
It is a novel framework, abbreviated from DEnoising diFfusion on continuous Tokens. The proposed framework represents users and items as latent representations, and leverages the exceptional continuous-valued generation capability of diffusion models to operate within continuous domains by conditioning on the discrete-valued reasoning content of LLMs.

<img width="839" alt="1744286175269" src="https://github.com/user-attachments/assets/b79d3630-8859-44a9-a906-a3b0db94d215" />

### An example of Implementation

Notably, please enter your huggingface token in Line 446 of the "model_interface.py" to get the access liscence for the Llama-3.2 model.

1. **Go to the path of "code"**
```
python cd code
```

2. **Training**
```
python main.py --dataset=software
```

3. **Evaluation**
```
python main.py --dataset=software --test_only
```

More configurations can be found in the "parse.py" file.

You can download the checkpoints at [Google Drive](https://drive.google.com/drive/folders/1CEZyrFMcTbCz2LO8Tm6L1oF9VPqqy9-K?usp=drive_link), and put them in the path of "ckpt/".
