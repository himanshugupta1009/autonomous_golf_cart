class Human:

    def __init__(self, state, goal, path):
        self.current_state = state
        self.goal = goal
        self.belief_goal = None
        self.path_to_goal = path

    def get_state(self):
        return self.current_state

    def get_goal(self):
        return self.goal

    def get_belief_goal(self):
        return self.belief_goal

    def get_path_to_goal(self):
        return self.path_to_goal

