import argparse
import random

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
import numpy as np

from rl.goal_value_net import GCVNetwork
from rl.traj_buffer import TrajectoryBufferDataset


def main(args):

    device = torch.device(args.device)
    dataset_path = args.dataset_path
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size

    dataset = torch.load(dataset_path)
    data_len = len(dataset)
    states, goal_values = dataset.states, dataset.goal_values

    train_dataset = TrajectoryBufferDataset(states=states[:int(0.9 * data_len)], goal_values=goal_values[:int(0.9 * data_len)])
    test_dataset = TrajectoryBufferDataset(states=states[-int(0.1 * data_len):], goal_values=goal_values[-int(0.1 * data_len):])
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = GCVNetwork(input_dim=124).to(device)
    
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_epoch():
        
        total_loss = 0.0

        for idx, data in enumerate(train_dataloader):
            
            state, label = data
            state = state.to(device).float()
            label = label.to(device).float()

            optimizer.zero_grad()
            pred = model(state).reshape(1, -1)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / (idx + 1)
    
    best_loss = np.inf
    for epoch in range(num_epochs):

        model.train()
        train_loss = train_epoch()
        
        model.eval()
        eval_loss = 0.0
        for idx, data in enumerate(test_dataloader):
            
            state, label = data
            state = state.to(device).float()
            label = label.to(device).float()

            pred = model(state).reshape(1, -1)
            loss = loss_fn(pred, label)
            
            eval_loss += loss.item()
        
        eval_loss /= (idx + 1)
        
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), './models/goal-conditioned/gcvf.pth')

        print('[EPOCH][{}][TRAIN LOSS][{}][EVAL LOSS][{}][BEST LOSS][{}]'.format(epoch, train_loss, eval_loss, best_loss))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset_path', type=str, default='datasets/traj_dataset.pt')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--num_epochs', type=int, default=1000)
    args = parser.parse_args()

    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    main(args)