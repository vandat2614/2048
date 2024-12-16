import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvBlock, self).__init__()
        d = output_dim // 4
        self.conv1 = nn.Conv2d(input_dim, d, 1, padding='same')
        self.conv2 = nn.Conv2d(input_dim, d, 2, padding='same')
        self.conv3 = nn.Conv2d(input_dim, d, 3, padding='same')
        self.conv4 = nn.Conv2d(input_dim, d, 4, padding='same')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        x = x.to(self.device)
        output1 = self.conv1(x)
        output2 = self.conv2(x)
        output3 = self.conv3(x)
        output4 = self.conv4(x)
        return torch.cat((output1, output2, output3, output4), dim=1)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = ConvBlock(16, 2048)
        self.conv2 = ConvBlock(2048, 2048)
        self.conv3 = ConvBlock(2048, 2048)
        self.dense1 = nn.Linear(2048 * 16, 1024)
        self.dense6 = nn.Linear(1024, 4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # Move model to the correct device

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = nn.Flatten()(x)
        x = F.dropout(self.dense1(x))
        return self.dense6(x)


class Agent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = DQN().to(self.device)
        self.target = DQN().to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.gamma = 0.9

    def encode_state(self, board):
        state = torch.tensor([0 if value == 0 else int(math.log(value, 2)) for value in board])
        state = F.one_hot(state, num_classes=16).float().flatten()
        state = state.reshape(1, 4, 4, 16).permute(0, 3, 1, 2)
        return state.to(self.device)  # Ensure the tensor is on the correct device

    def select_action(self, state):  # (1, 16, 4, 4)
        state_ = self.encode_state(state)
        return self.policy(state_).argmax(dim=-1).item()

    def update_model(self, states, actions, rewards, next_states, dones):
        for i in range(len(states)):
            states[i] = self.encode_state(states[i])
            next_states[i] = self.encode_state(next_states[i])

        states = torch.cat(states).to(self.device)
        next_states = torch.cat(next_states).to(self.device)

        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)

        current_q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(current_q_values, target_q_values)
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target.load_state_dict(self.policy.state_dict())
