# guardrail can allow or not allow an action.
# each action is a coin
# Here, states S = { 0, 1 }
# a = { null, act } 
# 

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

np.set_printoptions(precision=3)

N = 11 # number of possible theories
M = 500 # number of monte carlo samples 
T = 10 # number of timesteps

Pr = np.linspace(0,1,N)
Ut = np.array([[0,0], [-5, +1] ]) # (act?, outcome) -> utility
reg = Ut.max(axis=0, keepdims=True) - Ut
print('regret: ', reg)
# A = np.ones(N) # weights for theories

Amap = np.zeros((N, N, T))
Actprob = np.zeros((N, 3, T))
Us = np.zeros((N, 3, T))

# def regret(o, options):
#     return options.max() - o

# run experiments now
for c in range(N): # for each possible choice of true theory
    # if c != 3: continue
    print('\n\n ***** coin probability: ', Pr[c], '****')

    for m in range(M):
        A = np.ones(N) # weights for theories; initialize uniform
        U = np.zeros(3) # [ utility using expected utility decision, utility using mmregret ]
        samples = np.random.binomial(1, Pr[c], T)

        for t in range(T):
            ################## EU #######################
            # calculate expected utility, flatting each theory
            p1 = ((A/A.sum()) * Pr).sum() # combined probability of heads (good)
            eus = Ut[:,0] * (1-p1)  +  Ut[:,1] * (p1)

            # print('[t=%d] expected utilities'%t, eus)
            du = (1-samples[t]) * Ut[1,0] + (samples[t]) * Ut[1, 1]

            if eus[1] > eus[0]: # if positive EU, bet and incur some reward / loss.
                U[0] += du
                Actprob[c,0,t] += 1 / M
            


            ################ MWER #####################
            mwers = [
                max( A[j]* ( Pr[j] * reg[a,1]
                        + (1-Pr[j])* reg[a,0])
                    for j in range(N))
                for a in [0,1] ]
            # print('[t=%d] max expected regret'%t, options)
            # print('chosen from regrets')
            # print(np.array([
            #     [A[j]* ( Pr[j] * reg[a,1]
            #             + (1-Pr[j])* reg[a,0])
            #         for j in range(N)]
            #     for a in [0,1] ]))
            
            # if regret of action (action 1) is smaller in the worst case, then act:
            if mwers[1] < mwers[0]:
                U[1] += du
                Actprob[c,1,t] += 1 / M

            ################ MER #####################
            mers = [
                max( ( Pr[j] * reg[a,1]
                        + (1-Pr[j])* reg[a,0])
                    for j in range(N) if A[j] > 0)
                for a in [0,1] ]
            if mers[1] < mers[0]:
                U[2] += du
                Actprob[c,2,t] += 1 / M

            Us[c,:,t] += U / M


            # Now that utilities have been obtained, do learning.
            # the next two lines are equivalent:
            # A *= (Pr if samples[t]==1 else (1-Pr))  
            A = A * (Pr * samples[t] + (1-Pr)*(1-samples[t]))
            A /= A.max() # renormalize to max = 1
        
            Amap[c,:,t] += np.array(A) / M
        

curr_pos = 0

def update(key_evt):
    global curr_pos

    if key_evt:
        e = key_evt 
        curr_pos = (curr_pos + int(e.key=="right")-int(e.key=='left')) % N
    
    # ax.cla() # clear axis
    plt.clf()
    fig = plt.gcf()
    fig.suptitle("coin %d --- P(good) = %.2f" % (curr_pos, Pr[curr_pos]), fontweight='bold')
    ((ax1, ax2), (ax3, ax4)) = fig.subplots(2,2)

    sns.heatmap(np.flip(Amap[curr_pos,:,:],axis=0),vmin=0,vmax=1, ax=ax1)
    ax1.set_xlabel("t")
    ax1.set_ylabel("c")
    ax1.set_yticklabels(reversed(['1' if p >= 1 else '.%d'% int((p%1)*10) for p in Pr]))
    ax1.set_title("theory weights \\alpha")

    ax2.set_title("average total decision utility")
    ax2.plot(list(range(T)), Us[curr_pos,0], label='eu')
    ax2.plot(list(range(T)), Us[curr_pos,1], label='mwer')
    ax2.plot(list(range(T)), Us[curr_pos,2], label='mer')
    ax2.legend()

    ax3.set_title("probability of action")
    ax3.plot(list(range(T)), Actprob[curr_pos,0], label='eu')
    ax3.plot(list(range(T)), Actprob[curr_pos,1], label='mwer')
    ax3.plot(list(range(T)), Actprob[curr_pos,2], label='mer')
    ax3.legend()

    # ax4.cla()
    ax4.set_axis_off()

    fig.canvas.draw()


fig = plt.figure(figsize=(10,8))
fig.canvas.mpl_connect('key_press_event', update)
update(None)
plt.show()
        
