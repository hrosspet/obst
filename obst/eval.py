class Eval():
    def __init__(self, world, agent, training_steps, test_steps):
        self.world = world
        self.agent = agent
        self.training_steps = training_steps
        self.test_steps = test_steps

        self.training_score = 0
        self.test_score = 0

    def reset(self):
        self.world.reset()
        self.agent.reset()

    def eval(self, n_steps):
        self.reset()

        score = 0

        observation = self.world.state.observation
        for i in range(n_steps):
            # get agent's action based on the world observation
            action = self.agent.behave(observation)

            # get world's reaction
            observation, reward, done, _ = self.world.step(action)

            score += reward

            if done:
                self.reset()

        return score

    def train(self):
        self.training_score = self.eval(self.training_steps)
        return self.training_score / self.training_steps

    def test(self):
        self.test_score = self.eval(self.test_steps)
        return self.test_score / self.test_steps
