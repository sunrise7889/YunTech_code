import gymnasium as gym
import pygame
import sys
import time
import numpy as np
from hockey_env import AirHockeyEnv

def main():
    print("程式開始執行...")
    env = None
    
    try:
        # 註冊環境
        gym.register(
            id='AirHockey-v0',
            entry_point='hockey_env:AirHockeyEnv',
            max_episode_steps=1000,
        )
        print("環境註冊成功")
        
        # 初始化 Pygame
        pygame.init()
        print("Pygame 初始化成功")
        
        # 創建環境
        env = gym.make('AirHockey-v0', render_mode='human')
        print("環境創建成功")
        
        # 印出動作空間資訊
        print(f"Action space: {env.action_space}")
        
        # 重置環境
        observation, info = env.reset()
        print(f"環境重置成功，observation: {observation}")
        print("開始遊戲循環")
        
        running = True
        clock = pygame.time.Clock()
        frame_count = 0
        
        while running:
            frame_count += 1
            
            # 處理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            try:
                # 為兩個玩家生成動作
                # 假設動作空間是 [-1, 1] 的範圍
                player1_action = np.random.uniform(-1, 1, size=2)  # x, y 方向的移動
                player2_action = np.random.uniform(-1, 1, size=2)  # x, y 方向的移動
                
                # 合併動作
                combined_action = np.concatenate([player1_action, player2_action])
                
                # 執行遊戲邏輯
                observation, reward, terminated, truncated, info = env.step(combined_action)
                
                if terminated or truncated:
                    observation, info = env.reset()
            
            except Exception as e:
                print(f"執行步驟時發生錯誤: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
            
            # 控制遊戲速度
            clock.tick(60)
            
            # 短暫延遲以確保可以看到遊戲進行
            time.sleep(0.01)
            
            # 更新 pygame 顯示
            pygame.display.flip()
    
    except Exception as e:
        print(f"發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("關閉環境...")
        if env is not None:
            env.close()
        pygame.quit()
        print("程式結束")

if __name__ == "__main__":
    main()