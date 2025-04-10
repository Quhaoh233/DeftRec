import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go iLLMRec")
    
    # 
    parser.add_argument('--model', type=str, default='llama', help="the used language model")
    parser.add_argument('--test_only', default=False, action='store_true')
    
    # ----------------- encoding ----------------
    parser.add_argument('--k', type=int, default=2, help='K-way')
    parser.add_argument('--n', type=int, default=2, help='n layers')
    parser.add_argument('--user_token', default=False, action='store_true')
    parser.add_argument('--latent_dim', default=2048, type=int, help='The latent dimension of the embeddings that are sent into LLMs.')

    # ---------------- diffusion -----------------------
    parser.add_argument('--alpha', type=float, default=4, help='The weight of diffusion loss')
    parser.add_argument('--n_steps', type=int, default=1000, help='n_steps')
    parser.add_argument('--margin', type=float, default=2)
    parser.add_argument('--conditioning', type=str, default='ave', help="[ave, mlp]")
    
    # --------------- general --------------------
    parser.add_argument('--cuda', type=str, default='0', help="the used cuda")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--shuffle', default=False, action='store_true')
    parser.add_argument('--eval', type=str, default='mse', help='[mse, inner, cos]')
    parser.add_argument('--sample', default=False, action='store_true')   
    parser.add_argument('--sample_n', type=int, default=1000, help='A quick implementation for large datasets.')
    
    # learning
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_scheduler', default='cosine', choices=['cosine'], type=str)
    parser.add_argument('--lr_decay_min_lr', default=1e-9, type=float)
    parser.add_argument('--lr_warmup_start_lr', default=1e-7, type=float)
    parser.add_argument('--auto_lr_find', default=False, action='store_true')
    parser.add_argument('--batch_size', default=9, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    
    # llm
    parser.add_argument('--max_input_length', default=1024, type=int)
    parser.add_argument('--max_gen_length', default=32, type=int)
    
    # checkpoint
    parser.add_argument('--ckpt', default=False, action='store_true')
    parser.add_argument('--ckpt_name', type=str)
    
    # --------------- recommendation -----------------------------
    parser.add_argument('--data_name', default='lastfm', type=str)    
    parser.add_argument('--rec_dim', default=512, type=int)  
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--pi', default=0.1, type=float)
    
    return parser.parse_args()