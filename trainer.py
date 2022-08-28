from torch.utils.data import DataLoader
from torch import nn
import torch
from tqdm import tqdm

from torch_util.callback import CallbackList, BaseLogger, Tensorboard
from torch_util.metrics import compare_psnr



def train(module, dataset, file_path):

    train_epoch = 200

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0)

    module.cuda()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)

    callbacks = CallbackList(callbacks=[
        Tensorboard(file_path=file_path, per_batch=1),
    ])

    callbacks.set_module(module)
    callbacks.set_params({
        "lr": 1e-4,
        'train_epoch': train_epoch
    })

    global_batch = 1
    for global_epoch in range(1, train_epoch):

        module.train()

        iter_ = tqdm(dataloader, desc='Train [%.3d/%.3d]' % (global_epoch, train_epoch), total=len(dataset) // 16)

        for i, train_data in enumerate(iter_):
            x_init, P, S, y, x = (i.cuda() for i in train_data)
            x_init = x_init.permute([0, 3, 1, 2]).contiguous()

            if module.name == 'DEQ':
                x_hat, forward_iter, forward_res = module(x_init, P, S, y)
            else:
                x_hat = module(x_init, P, S, y)

            if x_hat.shape[1] == 2:
                x_hat = x_hat.permute([0, 2, 3, 1]).contiguous()
            else:
                continue

            #loss = loss_fn(torch.view_as_real(x_hat), torch.view_as_real(x))
            loss = loss_fn(x_hat, x)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            module.zero_grad()

            x_hat = torch.abs(x_hat)
            x_hat = (x_hat - torch.min(x_hat)) / (torch.max(x_hat) - torch.min(x_hat))

            x = torch.abs(x)
            x = (x - torch.min(x)) / (torch.max(x) - torch.min(x))

            log_batch = {
                'loss': loss.item(),
                'psnr': compare_psnr(x_hat, x).item()
            }

            if module.name == 'DEQ':
                log_batch.update(
                    {
                        'forward_iter': forward_iter,
                        'forward_res': forward_res
                    }
                )

            callbacks.call_batch_end_hook(log_batch, global_batch)

            if i == 10:
                callbacks.call_epoch_end_hook(log={}, image={
                    'x_hat': x_hat.unsqueeze(1)
                }, epoch=global_epoch)

            global_batch += 1


if __name__ == '__main__':

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    from method import CNNBlock, DeepUnfolding, DEQ
    from dataset import MoDLDataset

    train(DeepUnfolding(5), MoDLDataset(), './experimental_results/DU/')
    #train(CNNBlock(), MoDLDataset(), './experimental_results/CNNBlock')
    #train(DEQ(), MoDLDataset(), './experimental_results/DEQ/')
