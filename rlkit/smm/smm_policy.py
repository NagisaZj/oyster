import numpy as np
import matplotlib.pyplot as plt

class hard_smm_point():
    def __init__(self):
        return


    def get_action(self,observation):
        if np.sqrt(observation[0]**2+observation[1]**2)<=0.95:
            action = np.array([-0.1,0],dtype=np.float32)
        elif observation[1]<0:
            action = -1 * observation
            action = np.clip(action,-0.1,0.1)
        else:
            theta = np.arccos(observation[0]/np.sqrt(observation[0]**2+observation[1]**2))
            action = np.array([np.cos(theta-np.pi/2),np.sin(theta-np.pi/2)],dtype=np.float32)*0.1

        return action,None


if __name__=="__main__":
    policy = hard_smm_point()
    state = np.zeros((2,),dtype=np.float32)
    state_rem = np.zeros((200,2),dtype=np.float32)
    plt.figure()
    for i in range(1000):
        #state_rem[i] = state
        action,_ = policy.get_action(state)
        state = state + action
        plt.scatter(state[0],state[1])
        print(state)

    plt.show()




