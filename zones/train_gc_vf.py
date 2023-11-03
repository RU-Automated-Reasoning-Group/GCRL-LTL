import argparse
import random

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import numpy as np

from rl.goal_value_net import GCVNetwork
from envs.utils import get_zone_vector


def main(args):

    device = torch.device(args.device)
    dataset_path = args.dataset_path
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size

    #dataset = torch.load(dataset_path)
    dataset = torch.load('./datasets/traj_dataset_save.pt')
    train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    model = GCVNetwork(input_dim=124).to(device)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    J = get_zone_vector()['J']
    W = get_zone_vector()['W']
    R = get_zone_vector()['R']
    Y = get_zone_vector()['Y']

    def train_epoch(epoch_index):
        
        total_loss = 0.0

        for _, data in enumerate(train_dataloader):
            
            state, label = data
            label [label == -1] = 1

            state = state.to(device).float()
            label = label.to(device).float()

            optimizer.zero_grad()
            loss = None
            for idx, g in enumerate([J, W, R, Y]):
                gs = torch.from_numpy(np.tile(g, (state.shape[0], 1))).to(device)
                g_state = torch.hstack([state, gs]).float()
                g_label = label[:, idx:idx + 1]
                g_pred = model(g_state)
                if not loss:
                    loss = loss_fn(g_pred, g_label)
                else:
                    loss += loss_fn(g_pred, g_label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / batch_size
    
    best_loss = -np.inf
    for epoch in range(num_epochs):

        model.train()
        train_loss = train_epoch(epoch_index=epoch)
        # NOTE: skip eval for now
        # model.eval()
        
        if train_loss < best_loss:
            best_loss = train_loss
            #torch.save(model.state_dict(), './models/gc_vf.pth')

        print('[EPOCH][{}][LOSS][{}][BEST LOSS][{}]'.format(epoch, train_loss, best_loss))
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset_path', type=str, default='datasets/traj_dataset_save.pt')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=1000)
    args = parser.parse_args()

    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    main(args)