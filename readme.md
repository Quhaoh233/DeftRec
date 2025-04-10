# DeftRec
It is a novel framework, abbreviated from DEnoising diFfusion on continuous Tokens. The proposed framework represents users and items as latent representations, and leverages the exceptional continuous-valued generation capability of diffusion models to operate within continuous domains by conditioning on the discrete-valued reasoning content of LLMs.

<img width="839" alt="1744286175269" src="https://github.com/user-attachments/assets/b79d3630-8859-44a9-a906-a3b0db94d215" />

### An example of Implementation

Please download the checkpoints at [Google Drive](TBA), and put them in the path of "ckpt/".

1. **Go to the path of "code"**
```
python cd code
```

2. **Whole Pipeline**
```
python main.py --dataset=lastfm
```

3. **Train from checkpoint (LLM)**
```
python main.py --dataset=lastfm --train_from_checkpoint
```

4. **Evaluation**
```
python main.py --dataset=lastfm --test_only
