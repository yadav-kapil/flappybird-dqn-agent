import gymnasium as gym
import flappy_bird_gymnasium
import pygame


def main():
    pygame.init()
    clock = pygame.time.Clock()

    env = gym.make("FlappyBird-v0", render_mode="human")
    state, info = env.reset()

    running = True
    while running:
        action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = 1

        state, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            state, info = env.reset()

        clock.tick(30)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
