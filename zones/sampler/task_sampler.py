import random
import copy


class TaskSampler:
    def __init__(self, task, aps):
        self.task = task
        self.aps = aps

    def sample(self):
        
        aps = copy.copy(self.aps)
        random.shuffle(aps)

        if self.task == 'avoid':
            task_info = random.choice([('!+ U (+ && (!+ U +))', 4), ('(!+) U +', 2)])
            sketch, num_ap = task_info
            aps = random.sample(aps, k=num_ap)
            for ap in aps:
                sketch = sketch.replace('+', ap.lower(), 1)
        
        elif self.task == 'chain':
            sketch, num_ap = 'F(+ && F(+ && F(+ && F+)))', 4
            for ap in aps:
                sketch = sketch.replace('+', ap.lower(), 1)

        return sketch
        