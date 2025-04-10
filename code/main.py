import sys
import lightning as L
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import parse
import torch
import utils
import vae
from model_interface import MInterface
from data_interface import MyDataModule


def load_callbacks(args):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='metric',
        mode='max',
        patience=10,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='metric',
        dirpath='../ckpt/' + args.data_name,
        filename='{epoch:02d}-{metric:.3f}',
        save_top_k=-1,
        mode='max',
        save_last=True,
        #train_time_interval=args.val_check_interval
        every_n_epochs=1
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='step'))
    return callbacks


def main(args):
    print(f"Welcome to DeftRec! Now, we are working on Dataset: {args.data_name}.")
    L.seed_everything(args.seed)
    
    callbacks = load_callbacks(args)
    
    model = MInterface(**vars(args))
    if args.ckpt:
        ckpt_path = '../ckpt/' + args.data_name + '/' + args.ckpt_name + '.ckpt'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print("load checkpoints from {}".format(ckpt_path))

    data_module = MyDataModule(args)
    trainer = pl.Trainer(devices=[int(args.cuda)], accelerator='cuda', max_epochs=args.max_epochs, logger=True, callbacks=callbacks)  # single device

    if args.test_only:
        pass
    else:
        trainer.fit(model=model, datamodule=data_module)  # train and valid
    trainer.test(model=model, datamodule=data_module)  # test


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parse.parse_args()
    main(args)
