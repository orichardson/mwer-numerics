# guardrail can allow or not allow an action.
# each action is a coin
# Here, states S = { 0, 1 }
# a = { null, act } 
# 

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

## implement decision theory modules

# def eu(probs, utils): #  prob for each state, utility for each outcome -> 
#     return (probs * utils).sum()

np.set_printoptions(precision=3)

N = 11 # number of possible theories
M = 5000 # number of monte carlo samples 
T = 20 # number of timesteps

Pr = np.linspace(0,1,N)
# Ut = np.array([[0,0], [-5, +1] ]) # (act?, outcome) -> utility
Ut = np.array([[0,0], [-5, +1] ]) # (act?, outcome) -> utility
reg = Ut.max(axis=0, keepdims=True) - Ut
print('regret: ', reg)
# A = np.ones(N) # weights for theories

Amap = np.zeros((N, N, T))  # true coin, theory, timestep -> weight
Actprob = np.zeros((N, 3, T))
Us = np.zeros((N, 3, T))

# tracking some extra data for debugging
# p1s = np.zeros((N, T))
# all_p1s = np.zeros((M,N,T))
# all_eu1s = np.zeros((M,N,T))
# import pandas as pd
# all_actions = np.zeros((M,N,3,T))
# all_coins = np.zeros((M,N,T))


# def regret(o, options):
#     return options.max() - o


# run experiments now
for c in range(N): # for each possible choice of true theory
    # if c != 3: continue
    print('\n\n ***** coin probability: ', Pr[c], '****')

    samples = np.random.binomial(1, Pr[c], (M,T))


    for m in range(M):
        A = np.ones(N) # weights for theories; initialize uniform
        U = np.zeros(3) # [ utility using expected utility decision, utility using mmregret ]
        np.random.shuffle(samples)

        for t in range(T):
            ################## EU #######################
            # calculate expected utility, flatting each theory
            p1 = ((A/A.sum()) * Pr).sum() # combined probability of heads (good)
            # p1s[c,t] += p1/M
            eus = Ut[:,0] * (1-p1)  +  Ut[:,1] * (p1)


            #debug: the following two lines are equivalent
            samp = int(samples[m,t])
            # samp = int(np.random.rand() < Pr[c])
            # all_coins[m,c,t] = samp

            # if m == 1:
            #     print('[t=%d]; eus = '%t, eus)

            # print('[t=%d] expected utilities'%t, eus)
            # du = (1-samp) * Ut[1,0] + (samp) * Ut[1, 1]
            du = Ut[1,samp]

            eu_act = (eus[1] > eus[0])
            if eu_act: # if positive EU, bet and incur some reward / loss.
                U[0] += du
                Actprob[c,0,t] += 1.0 / M

            # all_actions[m,c,0,t] = float(eu_act)
            # all_eu1s[m,c,t] += eus[1]
            # all_p1s[m,c,t] = p1


            ################ MWER #####################
            mwers = [
                max( A[j]* ( Pr[j] * reg[a,1]
                        + (1-Pr[j]) * reg[a,0])
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
            mwer_act = (mwers[1] < mwers[0])
            if mwer_act:
                U[1] += du
                Actprob[c,1,t] += 1.0 / M
            # all_actions[m,c,1,t] = float(mwer_act)

            # ################ MER #####################
            mers = [
                max( ( Pr[j] * reg[a,1]
                        + (1-Pr[j])* reg[a,0])
                    for j in range(N) if A[j] > 0)
                for a in [0,1] ]
            
            mer_act = (mers[1] < mers[0])
            if mers[1] < mers[0]:
                U[2] += du
                Actprob[c,2,t] += 1.0 / M
            # all_actions[m,c,2,t] = mer_act

            Us[c,:,t] += np.array(U) / M


            # Now that utilities have been obtained, do learning.
            # the next two lines are equivalent:
            # A *= (Pr if samples[t]==1 else (1-Pr))  
            # A = A * (Pr * samples[t] + (1-Pr)*(1-samples[t]))
            # A = A * (Pr * samp + (1-Pr)*(1-samp))
            A *= Pr if samp==1 else (1-Pr)
            A /= A.max() # renormalize to max = 1
        
            Amap[c,:,t] += np.array(A) / M
        
# some postprocessing
# indices = np.indices(all_actions.shape).reshape(len(all_actions.shape), -1).T
# df = pd.DataFrame(indices,columns=['m', 'coin', 'decisionrule', 't'])
# df['action'] = all_actions.ravel()

# ind2 = np.indices(all_eu1s.shape).reshape(len(all_eu1s.shape), -1).T
# df2 = pd.DataFrame(ind2, columns=['m','coin', 't'])
# df2['outcomes'] = all_coins.ravel()
# df2['eu1'] = all_eu1s.ravel()
# df2['p1'] = all_p1s.ravel()

curr_pos = N//2

def update(key_evt):
    global curr_pos

    if key_evt:
        e = key_evt 
        curr_pos = (curr_pos + int(e.key=="right")-int(e.key=='left')) % N
    
    # ax.cla() # clear axis
    plt.clf()
    fig = plt.gcf()
    fig.suptitle("coin %d --- P(heads) = %.2f" % (curr_pos, Pr[curr_pos]), fontweight='bold')
    # ((ax1, ax2), (ax3, ax4), (ax5,ax6)) = fig.subplots(3,2)
    ((ax1, ax2), (ax3, ax4)) = fig.subplots(2,2)

    sns.heatmap(np.flip(Amap[curr_pos,:,:],axis=0),vmin=0,vmax=1, ax=ax1)
    ax1.set_xlabel("t")
    ax1.set_ylabel("c")
    # ax1.set_yticklabels(reversed(['1' if p >= 1 else '.%d'% int((p%1)*10) for p in Pr]))
    # print("number of tick labels: ", len(([('%.2f'%p)[1:] if p < 1 else '1' for p in Pr])))
    # ax1.set_yticklabels(reversed([('%.2f'%p)[1:] if p < 1 else '1' for p in Pr]))
    # ax1.set_ylim(0, 1)
    ax1.set_title("theory weights \\alpha")

    ax2.set_title("average total decision utility")
    Ts = list(range(T))
    ax2.plot(Ts, Us[curr_pos,0], label='eu')
    ax2.plot(Ts, Us[curr_pos,1], label='mwer')
    ax2.plot(Ts, Us[curr_pos,2], label='mer')
    ax2.legend()

    ax3.set_title("probability of action")
    ax3.plot(Ts, Actprob[curr_pos,0], label='eu')
    ax3.plot(Ts, Actprob[curr_pos,1], label='mwer')
    ax3.plot(Ts, Actprob[curr_pos,2], label='mer')
    ax3.legend()
    # sns.lineplot(df[df.coin==curr_pos],x='t',y='action', ax=ax5,
    #              hue='decisionrule', 
    #                 # estimator='mean', errorbar='sd',
    #                 # errorbar=('pi',90),
    #                 palette='bright')

    ax4.cla()
    ax4.set_axis_off()
    # ax4.plot(Ts, all_actions.sum(axis=0)[curr_pos,0,:]/M, label='P(action) -- eu')
    # ax4.plot(Ts, all_actions.sum(axis=0)[curr_pos,1,:]/M, label='P(action) -- mwer')
    # ax4.plot(Ts, p1s[curr_pos,:], label="P(heads)"),
    # ax4.legend()
    # ax4.set_ylim([0,1])

    # ax6.plot(Ts, all_eu1s[curr_pos,:], label="EU[act]")
    # ax6.legend()
    #
    # sns.lineplot(df2[df2.coin==curr_pos], x='t', y='p1', ax=ax6, errorbar=('pi',90), color='r')
    # ax62 = plt.twinx()
    # sns.lineplot(df2[df2.coin==curr_pos], x='t', y='eu1', ax=ax62,errorbar=('pi',90), color='g')

    fig.canvas.draw()


def show_plots():
    fig = plt.figure(figsize=(10,12))
    fig.canvas.mpl_connect('key_press_event', update)
    update(None)
    plt.show()
        

show_plots()