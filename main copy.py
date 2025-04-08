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
import json

from collections import deque
from utils.pause_train import pause_train
from utils.screen_capture import screen_capture
from utils.player_hp_detector import L_player_HP, R_player_HP
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
    
# DQN Agent with Combo System
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
        
        # Action sequence history for basic tracking
        self.action_history = []
        self.max_history_len = 5
        
        # Combo system
        self.combo_memory = []  # 存儲發現的有效連招
        self.current_combo = []  # 當前動作序列
        self.current_combo_rewards = []  # 當前連招的獎勵序列
        self.combo_reward_scale = 2.0  # 連招獎勵倍數
        self.combo_window = 12  # 連招識別窗口（步數）
        self.min_combo_length = 3  # 最小連招長度
        self.combo_cooldown = 0  # 連招識別冷卻
        self.combo_success = False  # 標記連招是否成功
        self.combo_threshold = 0.5  # 連招識別的獎勵閾值
        
        self.steps_done = 0
        self.episodes_done = 0
        
    def select_action(self, state):
        # 隨機使用已知連招
        if self.combo_memory and np.random.rand() < 0.3:  # 30%的機率使用已知連招
            chosen_combo = random.choice(self.combo_memory)
            if len(chosen_combo) > 0:
                combo_position = np.random.randint(0, len(chosen_combo))
                return chosen_combo[combo_position]
        
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
    
    def identify_combo(self, action, reward, done):
        """識別並處理連招"""
        # 記錄當前動作及其獎勵
        self.current_combo.append(action)
        self.current_combo_rewards.append(reward)
        
        # 保持連招窗口大小
        if len(self.current_combo) > self.combo_window:
            self.current_combo.pop(0)
            self.current_combo_rewards.pop(0)
        
        # 如果有足夠的正面獎勵，檢查是否形成有效連招
        if (not done and  # 回合結束時不識別連招
            len(self.current_combo) >= self.min_combo_length and 
            reward > self.combo_threshold and 
            self.combo_cooldown == 0):
            
            # 計算當前連招的總獎勵
            total_combo_reward = sum(self.current_combo_rewards[-self.min_combo_length:])
            
            # 如果連招總獎勵足夠高，認為是有效連招
            if total_combo_reward > self.combo_threshold * self.min_combo_length:
                potential_combo = self.current_combo[-self.min_combo_length:]
                
                # 檢查是否已經存在此連招
                if not any(self._is_similar_combo(potential_combo, existing_combo) for existing_combo in self.combo_memory):
                    self.combo_memory.append(potential_combo)
                    print(f"新連招發現! 動作序列: {[self._get_action_name(a) for a in potential_combo]}")
                    self.combo_success = True
                    self.combo_cooldown = 20  # 設置冷卻時間
                    return self.combo_reward_scale * reward  # 增加獎勵
        
        # 降低冷卻計數
        if self.combo_cooldown > 0:
            self.combo_cooldown -= 1
            
        self.combo_success = False
        return reward
    
    def _is_similar_combo(self, combo1, combo2):
        """檢查兩個連招是否相似"""
        if abs(len(combo1) - len(combo2)) > 1:
            return False
            
        # 簡單方法：檢查重疊元素比例
        min_len = min(len(combo1), len(combo2))
        matches = sum(a == b for a, b in zip(combo1[:min_len], combo2[:min_len]))
        overlap_ratio = matches / min_len
        return overlap_ratio > 0.7
    
    def _get_action_name(self, action_idx):
        """返回動作的可讀名稱"""
        action_names = [
            "no_op", "light_attack", "medium_attack", "heavy_attack", "special_attack",
            "drive_parry", "drive_impact", "throw", "assist", "jump",
            "move_left", "move_right", "crouch", "down_left", "down_right",
            "up_left", "up_right", "stop_movement", "hadouken", "hadouken_OD",
            "shoryuken", "shoryuken_OD", "dragonlash_kick", "dragonlash_kick_OD",
            "jinrai_kick", "jinrai_kick_OD", "low_spinning_sweep", "quick_dash",
            "DOWN_SP", "DOWN_SP_OD", "tatsumaki_senpukyaku_OD", "thunder_kick", "jump_in_heavy"
        ]
        if 0 <= action_idx < len(action_names):
            return action_names[action_idx]
        return f"action_{action_idx}"
    
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
        
    def save_combos(self, path="models", tag=None):
        """保存發現的連招"""
        os.makedirs(path, exist_ok=True)
        
        tag_str = f"_{tag}" if tag else ""
        filename = f"combos_ep{self.episodes_done}{tag_str}.json"
        full_path = os.path.join(path, filename)
        
        # 將動作索引轉換為動作名稱再保存
        named_combos = []
        for combo in self.combo_memory:
            named_combo = [self._get_action_name(action) for action in combo]
            named_combos.append(named_combo)
        
        with open(full_path, 'w') as f:
            json.dump(named_combos, f, indent=2)
        
        print(f"保存了 {len(self.combo_memory)} 個連招")

    def load_combos(self, path):
        """載入保存的連招"""
        if not os.path.exists(path):
            print(f"找不到連招文件: {path}")
            return
        
        with open(path, 'r') as f:
            named_combos = json.load(f)
        
        # 將動作名稱轉換回索引
        self.combo_memory = []
        action_names = [
            "no_op", "light_attack", "medium_attack", "heavy_attack", "special_attack",
            "drive_parry", "drive_impact", "throw", "assist", "jump",
            "move_left", "move_right", "crouch", "down_left", "down_right",
            "up_left", "up_right", "stop_movement", "hadouken", "hadouken_OD",
            "shoryuken", "shoryuken_OD", "dragonlash_kick", "dragonlash_kick_OD",
            "jinrai_kick", "jinrai_kick_OD", "low_spinning_sweep", "quick_dash",
            "DOWN_SP", "DOWN_SP_OD", "tatsumaki_senpukyaku_OD", "thunder_kick", "jump_in_heavy"
        ]
        
        for named_combo in named_combos:
            action_indices = []
            for name in named_combo:
                if name in action_names:
                    action_indices.append(action_names.index(name))
                else:
                    # 處理數字格式的動作
                    try:
                        action_idx = int(name.replace("action_", ""))
                        action_indices.append(action_idx)
                    except:
                        pass
            
            if action_indices:
                self.combo_memory.append(action_indices)
        
        print(f"載入了 {len(self.combo_memory)} 個連招")

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


# Street Fighter Environment wrapper with combo enhancement
class StreetFighterEnv:
    def __init__(self, screen_region=(0, 0, 1280, 720), attack_mode='modern'):
        self.screen = screen_capture(screen_region)
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
        
        # 連招系統
        self.combo_window = []  # 用於追蹤連招窗口內的動作和效果
        self.combo_timeout = 20  # 連招超時步數
        self.combo_active = False
        self.combo_counter = 0
        
        # self.player_HP_area = screen_capture((115, 85, 1165, 100))
        self.player_HP_area = self.screen[85:100, 115:1165]
        
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
        
        # 重置連招狀態
        self.combo_active = False
        self.combo_counter = 0
        self.combo_timeout = 0
        
        # Get initial state
        return self._get_state()
    
    def _get_state(self):
        # Process the image for the neural network
        # Convert to grayscale to reduce complexity
        gray = cv2.cvtColor(self.screen, cv2.COLOR_BGR2GRAY)
        
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
        # 記錄前HP
        previous_left_hp = self.last_left_hp
        previous_right_hp = self.last_right_hp
        
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
        
        # 計算此動作造成的傷害
        current_left_hp = L_player_HP(self.player_HP_area)
        current_right_hp = R_player_HP(self.player_HP_area)
        
        damage_dealt = previous_right_hp - current_right_hp
        damage_taken = previous_left_hp - current_left_hp
        
        # 添加到info
        info = {
            'damage_dealt': damage_dealt,
            'damage_taken': damage_taken,
            'combo_potential': damage_dealt > 3  # 有效攻擊判定
        }
        
        # Return step information
        return next_state, reward, done, info
    
    def _calculate_reward(self):
        # Get current HP values
        # player_HP_area = screen_capture((115, 85, 1165, 100))
        current_left_hp = L_player_HP(self.player_HP_area)
        current_right_hp = R_player_HP(self.player_HP_area)
        
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
            
            # 連招識別邏輯
            if not self.combo_active:
                self.combo_active = True
                self.combo_counter = 1
            else:
                self.combo_counter += 1
                # 連招獎勵遞增
                combo_bonus = min(self.combo_counter * 0.5, 3.0)
                reward += combo_bonus
                if self.combo_counter > 1:
                    print(f"連招 x{self.combo_counter}! 獎勵加成: +{combo_bonus:.1f}")
            
            # 重置連招計時器
            self.combo_timeout = 20
        
        # Penalty for taking damage
        if left_hp_change < 0:
            reward -= abs(left_hp_change) * 0.8
            
        # Small penalty for time passing (encourages faster action)
        reward -= 0.1
        
        # 降低連招計時器
        if self.combo_timeout > 0:
            self.combo_timeout -= 1
        else:
            # 連招結束
            if self.combo_active and self.combo_counter > 1:
                print(f"連招結束，總計 {self.combo_counter} 次連擊")
            self.combo_active = False
            self.combo_counter = 0
        
        # Update last HP values
        self.last_left_hp = current_left_hp
        self.last_right_hp = current_right_hp
        
        return reward
    
    def _is_done(self):
        # Episode ends if either player reaches 0 HP or max steps reached
        # player_HP_area = screen_capture((115, 85, 1165, 100))
        left_hp = L_player_HP(self.player_HP_area)
        right_hp = R_player_HP(self.player_HP_area)
        
        return left_hp <= 0 or right_hp <= 0 or self.step_count >= self.max_steps
    
    def close(self):
        # Make sure to release all keys when closing the environment
        self.character.impl.stop_movement()
        

# Training function with combo awareness
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
        
        # 嘗試加載連招數據
        combo_files = [f for f in os.listdir(model_path) if f.startswith('combos_') and f.endswith('.json')]
        if combo_files:
            latest_combo = max(combo_files, key=lambda x: int(x.split('ep')[1].split('_')[0].replace('.json', '') if 'ep' in x else 0))
            agent.load_combos(os.path.join(model_path, latest_combo))
    
    # Track pause state
    paused_train = False
    
    # 追蹤發現的連招
    discovered_combos_count = len(agent.combo_memory)
    
    try:
        for episode in range(agent.episodes_done + 1, episodes + 1):
            state = env.reset()
            total_reward = 0
            done = False
            step = 0
            episode_combos_found = 0
            
            print(f"\n第 {episode}/{episodes} 回合")
            
            # 運行一個回合
            while not done:
                # 檢查是否暫停
                paused_train = pause_train(paused_train, env)
                
                # 選擇並執行動作
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                # 連招識別和獎勵調整
                combo_reward = agent.identify_combo(action, reward, done)
                if agent.combo_success:
                    episode_combos_found += 1
                    reward = combo_reward  # 使用連招獎勵
                
                # 存儲轉換到重放緩衝區
                agent.memory.push(state, action, reward, next_state, done)
                
                # 訓練網絡
                agent.train()
                
                # 移動到下一個狀態
                state = next_state
                total_reward += reward
            
                step += 1
            
                # 打印狀態
                if step % 100 == 0:
                    combo_status = f", 連招數: {len(agent.combo_memory)}" if agent.combo_memory else ""
                    print(f"\r第 {step} 步, Epsilon: {agent.epsilon:.3f}, 獎勵: {total_reward:.2f}{combo_status}", end='')
                
                # 檢查是否手動中斷
                if keyboard.is_pressed('q'):
                    print("\n訓練被手動中斷")
                    agent.save_model()
                    agent.save_combos()
                    env.close()
                    return agent
                
            # 定期更新目標網絡
            if episode % target_update == 0:
                agent.update_target_network()
                
            # 打印回合統計信息
            combo_report = f", 本回合發現新連招: {episode_combos_found}" if episode_combos_found > 0 else ""
            total_combos = f", 總連招數: {len(agent.combo_memory)}" if agent.combo_memory else ""
            print(f"\r第 {episode} 回合結束，共 {step} 步。總獎勵: {total_reward:.2f}{combo_report}{total_combos}")
            agent.episodes_done = episode
            
            # 檢查是否有新的連招發現
            if len(agent.combo_memory) > discovered_combos_count:
                discovered_combos_count = len(agent.combo_memory)
                agent.save_combos()  # 發現新連招時保存
            
            # 保存檢查點
            if episode % save_interval == 0:
                agent.save_model()
                agent.save_combos(tag=f"interval_{episode}")
    except KeyboardInterrupt:
        print("\n訓練被用戶中斷")
        
    finally:
        # 確保在訓練結束時保存模型並釋放所有按鍵
        agent.save_model()
        agent.save_combos()
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

if __name__ == "__main__":
    time.sleep(3)
    
    # Choose mode: 'train' or 'test'
    mode = 'train'
    
    if mode == 'train':
        print("Starting DQN training for Street Fighter...")
        print("Press 'q' at any time to stop training")
        agent = train_agent(episodes=50000, target_update=10, save_interval=50)
    