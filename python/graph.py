#!/usr/bin/env python3

import time
import matplotlib.pyplot as plt

def main():
    while True:
        mean_loss = []
        mean_reward = []
        for line in open("./data/training_performance", "r"):
            if line != '\n':
                mean_loss.append(float(line.split(" ")[0]))
                mean_reward.append(float(line.split(" ")[1]))

        t = [i for i in range(len(mean_loss))]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        mean_loss_plot = fig.add_subplot(211)
        mean_reward_plot = fig.add_subplot(212)

        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

        mean_loss_plot.plot(t, mean_loss, color="blue")
        mean_reward_plot.plot(t, mean_reward, color="orange")

        mean_loss_plot.set_title("Mean Q-Value Loss")
        mean_reward_plot.set_title("Mean Reward")

        plt.pause(1)
        plt.close()

if __name__ == "__main__":
    
    main()
