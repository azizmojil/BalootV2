import multiprocessing as mp
from env.environment import BalootMultiAgentEnv

def worker(remote, parent_remote):
    parent_remote.close()
    env = BalootMultiAgentEnv()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, rewards, dones, infos = env.step(data)
            if dones.get('__all__', False):
                obs = env.reset()
            remote.send((obs, rewards, dones, infos, env.current_agent))
        elif cmd == 'reset':
            obs = env.reset()
            remote.send((obs, env.current_agent))
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError

class VectorEnv:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.processes = [mp.Process(target=worker, args=(work_remote, remote))
                          for work_remote, remote in zip(self.work_remotes, self.remotes)]
        for p in self.processes:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        return results

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        return results

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
