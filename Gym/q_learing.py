import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random

# 自定義桌上型曲棍球環境
class AirHockeyEnv(gym.Env):
    def __init__(self):
        super(AirHockeyEnv, self).__init__()

        # 桌面尺寸（單位：像素，對應小型桌上曲棍球）
        self.table_length = 500  # 桌台長度
        self.table_width = 250   # 桌台寬度
        self.goal_width = 80     # 球門寬度
        self.paddle_radius = 20  # 球拍半徑
        self.ball_radius = 10    # 球半徑
        self.max_speed = 15      # 球的最大速度

        # 定義觀察空間 (球的位置和速度，兩個球拍的位置)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -self.max_speed, -self.max_speed, 0, 0, self.table_length / 2, 0]),
            high=np.array([self.table_length, self.table_width, self.max_speed, self.max_speed, self.table_length / 2, self.table_width, self.table_length, self.table_width]),
            dtype=np.float32
        )

        # 定義動作空間 (5 種動作：靜止、上下左右移動)
        self.action_space = spaces.Discrete(5)

        self.reset()

    def reset(self):
        self.ball_pos = np.array([self.table_length / 2, self.table_width / 2])  # 球初始位置
        self.ball_vel = np.array([random.choice([-5, 5]), random.uniform(-3, 3)])  # 給球一個隨機初始速度
        
        # 修改球拍的初始位置，遠離球的位置
        self.paddle1_pos = np.array([self.table_length / 4, random.uniform(50, self.table_width - 50)])
        self.paddle2_pos = np.array([3 * self.table_length / 4, random.uniform(50, self.table_width - 50)])
        
        self.steps = 0
        self.max_steps = 500  # 每回合最大步數

        state = np.concatenate((self.ball_pos, self.ball_vel, self.paddle1_pos, self.paddle2_pos))
        return state, {}


    def step(self, action1, action2):
        action_map = {
            0: (0, 0),   # 靜止
            1: (-10, 0), # 向左
            2: (10, 0),  # 向右
            3: (0, -10), # 向上
            4: (0, 10)   # 向下
        }

        # 更新球拍位置
        dx1, dy1 = action_map[action1]
        dx2, dy2 = action_map[action2]
        self.paddle1_pos += np.array([dx1, dy1])
        self.paddle2_pos += np.array([dx2, dy2])

        # 限制球拍移動範圍
        self.paddle1_pos = np.clip(self.paddle1_pos, [0, 0], [self.table_length / 2, self.table_width])
        self.paddle2_pos = np.clip(self.paddle2_pos, [self.table_length / 2, 0], [self.table_length, self.table_width])

        # 球等待球拍擊球
        if np.linalg.norm(self.ball_pos - self.paddle1_pos) < self.ball_radius + self.paddle_radius:
            self.ball_vel = np.array([abs(self.ball_vel[0]) + 5, random.uniform(-5, 5)])
        elif np.linalg.norm(self.ball_pos - self.paddle2_pos) < self.ball_radius + self.paddle_radius:
            self.ball_vel = np.array([-abs(self.ball_vel[0]) - 5, random.uniform(-5, 5)])

        # 更新球位置
        self.ball_pos += self.ball_vel

        # 邊界碰撞檢測（上下反彈）
        if self.ball_pos[1] <= self.ball_radius or self.ball_pos[1] >= self.table_width - self.ball_radius:
            self.ball_vel[1] *= -1

        # 球門得分判定
        reward1, reward2 = 0, 0
        done = False
        if self.ball_pos[0] <= self.ball_radius and abs(self.ball_pos[1] - self.table_width / 2) <= self.goal_width / 2:
            reward2 = 1
            done = True
        elif self.ball_pos[0] >= self.table_length - self.ball_radius and abs(self.ball_pos[1] - self.table_width / 2) <= self.goal_width / 2:
            reward1 = 1
            done = True

        # 限制最大步數
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        state = np.concatenate((self.ball_pos, self.ball_vel, self.paddle1_pos, self.paddle2_pos))
        return state, reward1, reward2, done, {}

    def render(self, mode='human'):
        if not hasattr(self, 'screen') or self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.table_length, self.table_width))
            pygame.display.set_caption('Air Hockey')
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((0, 0, 0))  # 黑色背景

        # 繪製球
        ball_color = (255, 255, 255)  # 白色
        pygame.draw.circle(self.screen, ball_color, (int(self.ball_pos[0]), int(self.ball_pos[1])), self.ball_radius)

        # 繪製球拍1
        paddle1_color = (0, 255, 0)  # 綠色
        pygame.draw.circle(self.screen, paddle1_color, (int(self.paddle1_pos[0]), int(self.paddle1_pos[1])), self.paddle_radius)

        # 繪製球拍2
        paddle2_color = (255, 0, 0)  # 紅色
        pygame.draw.circle(self.screen, paddle2_color, (int(self.paddle2_pos[0]), int(self.paddle2_pos[1])), self.paddle_radius)

        # 更新畫面
        pygame.display.flip()
        self.clock.tick(30)  # 控制更新頻率

# Q-Learning 代理
class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def discretize_state(self, state):
        return tuple(np.round(state, 1))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space.n - 1)
        state_key = self.discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state, done):
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space.n)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space.n)

        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.discount_factor * self.q_table[next_state_key][best_next_action] * (not done)
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.learning_rate * td_error

        self.epsilon *= self.epsilon_decay

# 主程序
if __name__ == "__main__":
    env = AirHockeyEnv()
    agent1 = QLearningAgent(env.observation_space, env.action_space)
    agent2 = QLearningAgent(env.observation_space, env.action_space)

    # 訓練
    episodes = 500
    for episode in range(episodes):
        state, _ = env.reset()

        while True:
            # 球拍1嘗試朝球移動
            if state[4] < state[0]:
                action1 = 2  # 向右
            elif state[4] > state[0]:
                action1 = 1  # 向左
            elif state[5] < state[1]:
                action1 = 4  # 向下
            else:
                action1 = 3  # 向上

            # 球拍2嘗試朝球移動
            if state[6] < state[0]:
                action2 = 2  # 向右
            elif state[6] > state[0]:
                action2 = 1  # 向左
            elif state[7] < state[1]:
                action2 = 4  # 向下
            else:
                action2 = 3  # 向上

            next_state, reward1, reward2, done, _ = env.step(action1, action2)

            agent1.update(state, action1, reward1, next_state, done)
            agent2.update(state, action2, reward2, next_state, done)

            state = next_state

            if done:
                print(f"Episode {episode + 1} ended with reward1: {reward1}, reward2: {reward2}")
                break

    # 測試 AI
    test_episodes = 5
    for episode in range(test_episodes):
        state, _ = env.reset()
        print(f"Test Episode {episode + 1}")

        while True:
            # 球拍1嘗試朝球移動
            if state[4] < state[0]:
                action1 = 2  # 向右
            elif state[4] > state[0]:
                action1 = 1  # 向左
            elif state[5] < state[1]:
                action1 = 4  # 向下
            else:
                action1 = 3  # 向上

            # 球拍2嘗試朝球移動
            if state[6] < state[0]:
                action2 = 2  # 向右
            elif state[6] > state[0]:
                action2 = 1  # 向左
            elif state[7] < state[1]:
                action2 = 4  # 向下
            else:
                action2 = 3  # 向上

            next_state, _, _, done, _ = env.step(action1, action2)

            env.render()

            state = next_state

            if done:
                print(f"Test Episode {episode + 1} ended.")
                break
