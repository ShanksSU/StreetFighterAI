# main.py
import os
import random
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2

import time
import keyboard
from collections import deque
from controls.pause_train import pause_train
from image_processing.screen_capture import screen_capture
from image_processing.player_hp_detector import L_player_HP, R_player_HP
from characters.ken import Ken

class DQN(nn.Module):
    def __init__(self, input_channels, input_height, input_width, num_actions):
        super(DQN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate flattened conv output size
        conv_output_size = self._get_conv_output_size(input_channels, input_height, input_width)

        # Value stream: output a single state value V(s)
        self.fc_value1 = nn.Linear(conv_output_size, 512)
        self.fc_value2 = nn.Linear(512, 1)

        # Advantage stream: output A(s, a) for each action
        self.fc_advantage1 = nn.Linear(conv_output_size, 512)
        self.fc_advantage2 = nn.Linear(512, num_actions)

    def _get_conv_output_size(self, input_channels, height, width):
        """Calculate the flattened size after convolution layers."""
        dummy_input = torch.zeros(1, input_channels, height, width)
        feature_1 = F.relu(self.conv1(dummy_input))
        feature_2 = F.relu(self.conv2(feature_1))
        feature_3 = F.relu(self.conv3(feature_2))
        return int(np.prod(feature_3.size()))

    def forward(self, input_image):
        # CNN feature extraction
        feature_1 = F.relu(self.conv1(input_image))
        feature_2 = F.relu(self.conv2(feature_1))
        feature_3 = F.relu(self.conv3(feature_2))
        flattened = feature_3.view(feature_3.size(0), -1)  # Flatten

        # Value stream
        value = F.relu(self.fc_value1(flattened))
        value = self.fc_value2(value)  # Shape: (batch, 1)

        # Advantage stream
        advantage = F.relu(self.fc_advantage1(flattened))
        advantage = self.fc_advantage2(advantage)  # Shape: (batch, num_actions)

        # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

    
# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def __len__(self):
        return len(self.buffer)
    
# DQN Agent
class DQNAgent:
    def __init__(self, input_channels, input_height, input_width, num_actions, 
                 lr=0.0001, gamma=0.99, epsilon=1.0, epsilon_min=0.1, 
                 epsilon_decay=0.995, buffer_size=10000, batch_size=32):
        
        self.num_actions = num_actions
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Q networks
        self.policy_net = DQN(input_channels, input_height, input_width, num_actions)
        self.target_net = DQN(input_channels, input_height, input_width, num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        
        # Set device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        
        # Action sequence history
        self.action_history = []
        self.max_history_len = 5
        
        self.steps_done = 0
        self.episodes_done = 0
        
    def select_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        
        # Convert state to PyTorch tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.max(1)[1].item()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q(s_t, a)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute V(s_{t+1})
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        
        # Compute expected Q values
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = F.smooth_l1_loss(q_values.squeeze(), expected_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients (optional)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save_model(self, path="models", tag=None, include_time=True):
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d") if include_time else ""

        tag_str = f"_{tag}" if tag else ""

        filename = f"sf_agent_ep{self.episodes_done}{tag_str}_{timestamp}.pth" if timestamp else f"sf_agent_ep{self.episodes_done}{tag_str}.pth"
        full_path = os.path.join(path, filename)

        # Save model
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'epsilon': self.epsilon
        }, full_path)

        print(f"Model saved at episode {self.episodes_done}")

    def load_model(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            
            try:
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.target_net.load_state_dict(checkpoint['target_net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.steps_done = checkpoint['steps_done']
                self.episodes_done = checkpoint['episodes_done']
                self.epsilon = checkpoint['epsilon']
                
                print(f"Model loaded from {checkpoint_path}")
            except RuntimeError as e:
                print(f"Error loading model: {e}")
                print("Detected architecture change. Creating new model...")
                # Just load the non-model parameters
                self.steps_done = checkpoint.get('steps_done', 0)
                self.episodes_done = checkpoint.get('episodes_done', 0)
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                print(f"Continued from episode {self.episodes_done} with epsilon {self.epsilon}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")
        
# Street Fighter Environment wrapper
class StreetFighterEnv:
    def __init__(self, screen_region=(0, 0, 1280, 720), attack_mode='modern'):
        self.screen_region = screen_region
        self.character = Ken(attack_mode=attack_mode)
        self.last_left_hp = 100
        self.last_right_hp = 100
        self.step_count = 0
        self.max_steps = 1000  # Maximum steps per episode
        
        # Movement states
        self.moving_left = False
        self.moving_right = False
        self.crouching = False
        
        # Define actions (map integers to game actions)
        self.actions = [
            self._no_op,
            # === Basic Attack ===
            self.character.impl.light_attack,
            self.character.impl.medium_attack,
            self.character.impl.heavy_attack,
            self.character.impl.special_attack,

            # === other commands ===
            self.character.impl.drive_parry,
            self.character.impl.drive_impact,
            self.character.impl.throw,
            self.character.impl.assist,
            
            # === Basic movement ===
            self.character.impl.move_jump,
            self.character.impl.move_left_continuously,
            self.character.impl.move_right_continuously,
            self.character.impl.move_crouch_continuously,
            self.character.impl.down_left_continuously,
            self.character.impl.down_right_continuously,
            self.character.impl.up_left_continuously,
            self.character.impl.up_right_continuously,
            self.character.impl.stop_movement,
            
            # === SP Attack ===
            self.character.hadouken,
            self.character.hadouken_OD,
            self.character.shoryuken,
            self.character.shoryuken_OD,
            self.character.dragonlash_kick,
            self.character.dragonlash_kick_OD,
            self.character.jinrai_kick,
            self.character.jinrai_kick_OD,
            
            self.character.low_spinning_sweep,
            self.character.quick_dash,
            self.character.DOWN_SP,
            self.character.DOWN_SP_OD,
            self.character.tatsumaki_senpukyaku_OD,
            self.character.thunder_kick,
            self.character.jump_in_heavy,
        ]
        
    def _no_op(self):
        # Do nothing action
        time.sleep(0.1)
        
    def reset(self):
        # Reset the game (you might need to implement a way to restart the match)
        # For now, we'll just wait a bit and assume the game is reset
        self.character.impl.stop_movement()
        time.sleep(3)
        self.step_count = 0
        self.last_left_hp = 100
        self.last_right_hp = 100
        
        # Get initial state
        return self._get_state()
    
    def _get_state(self):
        # Capture game screen
        screen = screen_capture(self.screen_region)
        
        # Process the image for the neural network
        # Convert to grayscale to reduce complexity
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        
        # Stack 4 frames for temporal information (optional, useful for detecting motion)
        # For simplicity, we're using just one frame here
        
        # Resize to manageable size
        resized = cv2.resize(gray, (84, 84))
        
        # Normalize pixel values
        normalized = resized / 255.0
        
        # Add channel dimension (DQN expects inputs in format [batch, channels, height, width])
        state = np.expand_dims(normalized, axis=0)
        
        return state
    
    def step(self, action_idx):
        # Execute the selected action
        if 0 <= action_idx < len(self.actions):
            self.actions[action_idx]()
        
        # Small delay to let the game process the action
        time.sleep(0.05)
        
        # Get new state
        next_state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        self.step_count += 1
        done = self._is_done()
        
        # Return step information
        return next_state, reward, done, {}
    
    def _calculate_reward(self):
        # Get current HP values
        player_HP_area = screen_capture((115, 85, 1165, 100))
        current_left_hp = L_player_HP(player_HP_area)
        current_right_hp = R_player_HP(player_HP_area)
        
        # Check player_HP_area
        # print(f"\rLeft HP: {current_left_hp}, Right HP: {current_right_hp}", end='')
        # cv2.imshow("123", player_HP_area) # this
        # cv2.waitKey(1)
        
        # Calculate HP changes
        left_hp_change = current_left_hp - self.last_left_hp
        right_hp_change = current_right_hp - self.last_right_hp
        
        # Assuming player is on the left
        reward = 0
        
        # Reward for dealing damage to opponent
        if right_hp_change < 0:
            reward += abs(right_hp_change) * 1.0  # More reward for dealing damage
        
        # Penalty for taking damage
        if left_hp_change < 0:
            reward -= abs(left_hp_change) * 0.8
            
        # Small penalty for time passing (encourages faster action)
        reward -= 0.1
        
        # Update last HP values
        self.last_left_hp = current_left_hp
        self.last_right_hp = current_right_hp
        
        return reward
    
    def _is_done(self):
        # Episode ends if either player reaches 0 HP or max steps reached
        player_HP_area = screen_capture((115, 85, 1165, 100))
        left_hp = L_player_HP(player_HP_area)
        right_hp = R_player_HP(player_HP_area)
        
        return left_hp <= 0 or right_hp <= 0 or self.step_count >= self.max_steps
    
    def close(self):
        # Make sure to release all keys when closing the environment
        self.character.stop_movement()
        
# Training function
def train_agent(episodes=50000, target_update=10, save_interval=50):
    # Define environment and agent parameters
    env = StreetFighterEnv(screen_region=(0, 0, 1280, 720), attack_mode='modern')
    input_channels = 1  # Grayscale image has 1 channel
    input_height = 84
    input_width = 84
    num_actions = len(env.actions)
    
    agent = DQNAgent(
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        num_actions=num_actions,
        lr=0.0001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=32
    )
    
    # Try to load previous checkpoint
    model_path = "models"
    checkpoint_files = [f for f in os.listdir(model_path) if f.endswith('.pth')] if os.path.exists(model_path) else []

    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('ep')[1].split('_')[0]) 
                            if 'ep' in x and x.split('ep')[1].split('_')[0].isdigit() else 0)
        print(f"Found latest checkpoint: {latest_checkpoint}")
        agent.load_model(os.path.join(model_path, latest_checkpoint))
    
    # Track pause state
    paused_train = False
    
    try:
        for episode in range(agent.episodes_done + 1, episodes + 1):
            state = env.reset()
            total_reward = 0
            done = False
            step = 0
            agent.action_history = []  # Reset action history at start of episode
            
            print(f"\nEpisode {episode}/{episodes}")
            
            # Run one episode
            while not done:
                # Check for pause
                paused_train = pause_train(paused_train)
                
                # Select and perform action
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Store transition in replay buffer
                agent.memory.push(state, action, reward, next_state, done)
                
                # Train the network
                agent.train()
                
                # Move to next state
                state = next_state
                total_reward += reward
            
                step += 1
            
                # Print status
                if step % 100 == 0:
                    print(f"\rStep {step}, Epsilon: {agent.epsilon:.3f}, Reward: {total_reward:.2f}", end='')
                
                # Check for manual interrupt
                if keyboard.is_pressed('q'):
                    print("\nTraining manually interrupted")
                    agent.save_model()
                    env.close()
                    return agent
                
            # Update target network periodically
            if episode % target_update == 0:
                agent.update_target_network()
                
            # Print episode stats
            print(f"\rEpisode {episode} finished after {step} steps. Total reward: {total_reward:.2f}")
            agent.episodes_done = episode
            
            # Save checkpoint
            if episode % save_interval == 0:
                agent.save_model()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    finally:
        # Make sure to save model and release all keys when training ends
        agent.save_model()
        env.close()
    
    return agent

# Test the trained agent
def test_agent(agent, episodes=5):
    env = StreetFighterEnv(screen_region=(0, 0, 1280, 720), attack_mode='modern')
    
    try:
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Always choose best action during testing (no exploration)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    q_values = agent.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
                
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward
                
                # Check for manual interrupt
                if keyboard.is_pressed('q'):
                    print("\nTesting manually interrupted")
                    return
            
            print(f"Test Episode {episode}, Total Reward: {total_reward:.2f}")
    
    finally:
        # Make sure to release all keys when testing ends
        env.close()

# Add this function to load a model for testing
def load_model_for_testing(model_path=None):
    # Define environment to get action space size
    env = StreetFighterEnv(screen_region=(0, 0, 1280, 720), attack_mode='modern')
    input_channels = 1  # Grayscale image has 1 channel
    input_height = 84
    input_width = 84
    num_actions = len(env.actions)
    
    # Initialize agent with the same parameters used during training
    agent = DQNAgent(
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        num_actions=num_actions,
        lr=0.0001,
        gamma=0.99,
        epsilon=0.1,  # Low epsilon for testing (minimal exploration)
        epsilon_min=0.1,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=32
    )
    
    # If specific model path is provided, load it
    if model_path and os.path.exists(model_path):
        agent.load_model(model_path)
        return agent
    
    # Otherwise find the latest checkpoint
    model_dir = "models"
    if os.path.exists(model_dir):
        checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if checkpoint_files:
            # Sort by episode number extracted from filename
            latest_checkpoint = max(checkpoint_files, 
                                   key=lambda x: int(x.split('ep')[1].split('_')[0]) 
                                   if 'ep' in x else 0)
            agent.load_model(os.path.join(model_dir, latest_checkpoint))
            return agent
    
    print("No model checkpoint found. Please train the agent first or provide a valid model path.")
    return None

if __name__ == "__main__":
    time.sleep(3)
    
    # Choose mode: 'train' or 'test'
    mode = 'train'
    
    if mode == 'train':
        print("Starting DQN training for Street Fighter...")
        print("Press 'q' at any time to stop training")
        agent = train_agent(episodes=50000, target_update=10, save_interval=50)
    
    elif mode == 'test':
        print("Loading model for testing...")
        # find the latest model automatically
        agent = load_model_for_testing()
        # agent = load_model_for_testing("models/sf_agent_ep500_20240407.pth")
        
        if agent:
            print("Testing trained agent...")
            # An episode is one full match
            test_agent(agent, episodes=5)