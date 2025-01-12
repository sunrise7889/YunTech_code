import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import sys

class AirHockeyEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode='human'):
        super().__init__()
        print("初始化環境...")
        
        # 初始化Pygame
        pygame.init()
        
        # 定義邊界座標和顏色
        self.upline = 85    
        self.downline = 415 
        self.screen_width = 640
        self.screen_height = 480
        
        # 顏色定義
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        
        print("設置顯示視窗...")
        # 設置顯示視窗
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Air Hockey')
        
        # 定義物理參數
        self.puck_radius = 15
        self.mallet_radius = 20
        self.max_velocity = 10.0
        self.friction = 0.95
        
        # 定義觀察和動作空間
        self.observation_space = spaces.Box(
            low=np.array([0, self.upline, -self.max_velocity, -self.max_velocity,
                         0, self.upline, 0, self.upline], dtype=np.float32),
            high=np.array([self.screen_width, self.downline, self.max_velocity, self.max_velocity,
                          self.screen_width, self.downline, self.screen_width, self.downline], dtype=np.float32),
        )
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )
        
        self.clock = pygame.time.Clock()
        self.render_mode = render_mode
        print("環境初始化完成")

    def reset(self, seed=None, options=None):
        print("重置環境...")
        super().reset(seed=seed)
        
        # 重置遊戲狀態
        self.puck_pos = np.array([self.screen_width/2, (self.upline + self.downline)/2], dtype=np.float32)
        self.puck_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.player_mallet = np.array([self.screen_width/4, (self.upline + self.downline)/2], dtype=np.float32)
        self.opponent_mallet = np.array([3*self.screen_width/4, (self.upline + self.downline)/2], dtype=np.float32)
        
        # 清空屏幕
        self.screen.fill(self.BLACK)
        
        # 渲染初始狀態
        self._render_frame()
        
        # 確保返回正確的格式
        observation = self._get_observation()
        info = {}
        print("重置完成")
        return observation, info

    def step(self, action):
        print("執行步驟...")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # 更新玩家推桿位置
        action = np.clip(action, -1.0, 1.0)
        self.player_mallet += action * 5
        
        # 確保推桿在合法範圍內
        self.player_mallet = np.clip(
            self.player_mallet,
            [self.mallet_radius, self.upline + self.mallet_radius],
            [self.screen_width/2 - self.mallet_radius, self.downline - self.mallet_radius]
        )
        
        # 更新對手推桿
        self.opponent_mallet[1] += (self.puck_pos[1] - self.opponent_mallet[1]) * 0.05
        self.opponent_mallet[1] = np.clip(
            self.opponent_mallet[1],
            self.upline + self.mallet_radius,
            self.downline - self.mallet_radius
        )
        
        # 更新冰球位置和速度
        self.puck_pos += self.puck_vel
        self.puck_vel *= self.friction
        
        self._check_collisions()
        
        # 渲染當前幀
        self._render_frame()
        
        # 限制幀率
        self.clock.tick(60)
        
        terminated = False
        reward = 0.0
        if (self.puck_pos[0] <= self.puck_radius or 
            self.puck_pos[0] >= self.screen_width - self.puck_radius):
            terminated = True
            reward = 1.0 if self.puck_pos[0] >= self.screen_width - self.puck_radius else -1.0
        
        observation = self._get_observation()
        info = {}
        print("步驟完成")
        return observation, reward, terminated, False, info

    def _render_frame(self):
        # 清空屏幕
        self.screen.fill(self.BLACK)
        
        # 繪製邊界
        pygame.draw.line(self.screen, self.WHITE, (0, self.upline), (self.screen_width, self.upline), 2)
        pygame.draw.line(self.screen, self.WHITE, (0, self.downline), (self.screen_width, self.downline), 2)
        
        # 繪製中線
        pygame.draw.line(self.screen, self.WHITE, (self.screen_width/2, self.upline), 
                        (self.screen_width/2, self.downline), 2)
        
        # 繪製冰球
        pygame.draw.circle(self.screen, self.WHITE, self.puck_pos.astype(int), self.puck_radius)
        
        # 繪製玩家推桿
        pygame.draw.circle(self.screen, self.RED, self.player_mallet.astype(int), self.mallet_radius)
        
        # 繪製對手推桿
        pygame.draw.circle(self.screen, self.BLUE, self.opponent_mallet.astype(int), self.mallet_radius)
        
        # 更新顯示
        pygame.display.flip()

    def _get_observation(self):
        return np.concatenate([
            self.puck_pos,
            self.puck_vel,
            self.player_mallet,
            self.opponent_mallet
        ]).astype(np.float32)

    def _check_collisions(self):
        # 檢查與邊界的碰撞
        if self.puck_pos[1] <= self.upline + self.puck_radius:
            self.puck_pos[1] = self.upline + self.puck_radius
            self.puck_vel[1] *= -0.8
        elif self.puck_pos[1] >= self.downline - self.puck_radius:
            self.puck_pos[1] = self.downline - self.puck_radius
            self.puck_vel[1] *= -0.8
        
        # 檢查與推桿的碰撞
        for mallet_pos in [self.player_mallet, self.opponent_mallet]:
            diff = self.puck_pos - mallet_pos
            dist = np.linalg.norm(diff)
            if dist < (self.puck_radius + self.mallet_radius):
                norm = diff / dist
                self.puck_vel = norm * self.max_velocity * 0.8
                self.puck_pos = mallet_pos + norm * (self.puck_radius + self.mallet_radius)

    def close(self):
        print("關閉環境...")
        pygame.quit()