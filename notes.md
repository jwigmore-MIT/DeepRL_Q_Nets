# TO RUN
1. With advantage normalization

## Why would the policy loss be much smaller for the shared optimizer?
Learning occurs rapidly when policy loss gets small

# Notes
## Advantage Formula
delta = rewards[t] + gamma*V(t+1)*I(t+1 is not terminal) - V(t)
advantages[t] = delta + (gamma * lambda * advantages[t+1]) * I(t+1 is not terminal)

## Value Function Bootstrapping
"If a sub-environment is not terminated nor truncated, PPO estimates the value of the next state in this sub-environment as the value target." [37]
(WRONG) We are technically truncating the environment because we are wrapping it in a time-limit wrapper. So we should NEVER bootstrap. 
(CORRECT) IF num_steps_per_env != reset_steps_per_env then we do bootstrap after every rollout
If done incorrectly, this can be bad.  If gamma = 1, and we bootstrap the final value, this bootstrapped estimate will be infinite.



## Training Step Terminology
train_steps = 1 step taken in all environments i.e. num_steps in a rollout 
global_step = sum of steps taken in all environments i.e. num_environments * train_steps \\ Not used
rollouts = 1 taken before each learning phase = 1 batch of data for the learning phase
resets = 1 taken every reset_steps/num_steps_per_rollout
updates = in each learning phase, update_per_rollouts are taken 
mb_updates = for each update, num_minibatches are taken 

batch_size = num_envs * num_steps_per_rollout
minibatch_size = batch_size // num_minibatches_per_update



## Understanding the training Loop

for update (epoch?) in range(num_updates+1): 
    \\ COLLECT ROLLOUT
    for step in range(num_steps_per_env):
        Collect trajectory of length num_steps_per_env for EACH parallel environment
        if trajectory is not finished: 
            continue
        else: 
            Compute the average episodic return over the parallel trajectories and log
            Call `checkpoint_saver` to see if the avg_LTA_return is one of the best of all runs, if so, save it            
            if average_LTA_return > previous best in current run:
                Manually save the agent (just in case)
                Log average_LTA_return as the new best
    \\ COMPUTE THE ADVANTAGES WITH BOOTSTRAPPING IF TRAJECTORIES DID NOT END WITH TERMINATION
    Compute the advantages for each step in each trajectory 
    returns[t] = A(s[t],a[t]) + V(s[t]) for all t  
    \\ PREP ROLLOUT INTO A "BATCH"
    Flatten all observations, log_probs, actions, advantages, returns, and values
    \\ LEARNING PHASE
    for epoch in range(update_epochs) \\ i.e. we update using each batch multiple times
        \\ Note: after the below loop is run once, we have a new actor and critic
        \\ and this loop will be run update_epochs*num_minibatches number of times
        for minibatch in range(num_minibatches) \\implementation looks different but this is essentially what we are doing
            Sample a random minibatch
            Pass minibatch observations and actions to the CURRENT actor and critic to compute log probabilities, entropies, and values 
            \\ Compute ratios and losses used to update actor
            Compute ratio = exp( log(pi_new) - log(pi_old)) \\ = pi_new/pi_old
            Approximate KL-divergence  \\ Mostly needed for logging
            Compute clipfracs += mean((|1-ratio| > clip_coef)) \\ percentage of pi_new(s) that deviates from the old be a factor greater than 1-epsilon 
            Compute losses based on the minibatch advantages, the ratio, and clipping   
            \\ Compute critic losses
            v_loss = compute_clip_v_loss()
                or
            v_loss = mean(1/2(V_new(s) - (A(s,a)+V_old(s)))^2) \\ to minimize MSE between V_new and Target (Advantage + V_old)
            \\ Compute entropy loss
            entropy_loss = entropy.mean()
            \\ Compute overall loss
            loss = + pg_loss \\ PPO clipped policy loss
                   - ent_coef * entropy loss \\ actually rewards entropy
                   + vf_coeff * v_loss \\ Value function loss
            \\ Update via backpropagation with gradient clipping
            optimizer.zero_grad()
            loss.backwards()
            nn.utils.clip_grad_norm_(agent.parameters(), train_args.max_grad_norm) 
    \\ Continue on after running above after performing update_epochs number of passes through the loop
    \\ meaning we use the num_steps*num_envs batch_size data update_epochs number of times
    \\ Computing metrics based on the error in the value estimates
    y_pred = b_values \\ critic estimates of the value for each observation in the batch
    y_true = b_returns \\ observed returns for each observation in the batch
    var_y = np.var(y_true)    
    diff_y = y_true - y_pred
    abs_diff_y = np.abs(diff_y)
    explained_var = { np.nan                            if var_y == 0
                      1-np.var(y_true-y_pred)/var_y     else


## Why is V_target = Advantage + V_old = (Q(s,a)-V_old)

## Returns = advantages + values
advantages = GAE(rewards, future_rewards, gamma, lambda) // advantage estimation WITHOUT critic networks involvement
values = critic network estimate
A(s_t,a_t) = Q(s_t,a_t)                       - V(s_t)
           = r(s_t, a_t) + gamma * V(s_{t+1}) - V(s_t)
           ~ \sum_{l}^{\infty} \gamma * \lamda \delta_{t+l}
where
    \delta_{t+l) =  l step estimate of the advantage i.e.
    \delta_{t+2} = r_t + \gamma * r_{t+1) + \gamma^2 + \gamma^2 * r_{t+2} + \gamma^3 V(s_{t+2}) - V(s_t) 

    
## Checkpoint Saver
Compares the avg_LTA_return with the best avg_LTA_returns obtained when training on the same problem instance
    

## Minibatches
Updates are based on "minibatches"
1. We collect a rollout of size `num_steps` for each environment
2. Flatten rollouts into one continuous vector
3. Sample minibatches of size train_args.minibatch_size randomly from this flattened batch vector
SO THE MINIBATCHES ARE NOT FROM A SINGLE TRAJECTORY OR CONTINUOUS
   

## Entropy Loss
In minibatch update loop:
    entropy = agent.get_action_and_value(batch_data)

Entropy loss = mean(entropy(minibatch)) 
    i.e. the mean is taken over the minibatch dimension and the entropy 
    is summed over the action_space dimension

SUMMARY: 
1. We take minibatch data
2. pass into actor network, 
3. get the sum of entropy for each observation in the minibatch (dim 1)
4. Take the mean of this summation over all observations in the minibatch (dim 0) to get the entropy loss

### Entropy from agent
Each agent.get_action_and_value() passes all the observations in the minibatch through the actor network
    Note: observation data is minibatch_size * state_space_size 
For each observation a distribution over possible actions is computed:
    action_mean = actor_mean(minibatch observations) \\ OUT: minibatch_size * state_space_size OUT: minibatch_size * action_space_size
    action_logstd = actor_logstd.expand_as(action_mean) \\ OUT: same dimensions as action_mean
    probs = Norma(action_mean, action_std) \\ minibatch dimensional MVN distribution with zero covariance
This is a multivariate normal distribution with a diagonal covariance matrix
Computing both the log_probability and entropy is a sum over the action_space dimension 
    meaning logprobs and entropy have the same size as the minibatch size


## Gradient Clipping
"For each update iteration in an epoch, PPO rescales the gradients of the policy and 
value network so that the “global l2 norm” (i.e., the norm of the concatenated gradients of 
all parameters) does not exceed 0.5."

## Testing
stabilized| gamma | time_scaled | moving_average | max_grad_norm | norm_adv |   notes
---------------------------------------------------------------------------------------
true      |  0.9  | true        | false          | 0.5           | true     | Modest value lost (starts at 275 ends at < 2)
false     |  0.9  | false       | true           | 0.5           | true     | Essentially zero value lost throughout
false     |  0.9  | false       | false          | 0.5           | true     | Very large value lost (6e6)
false     |  0.9  | false       | false          | none          | true     | Very large value lost (6e6)
false     |  0.9  | false       | false          | 0.5           | false    | Very large value loss (6e6)




## Debug Test
1. Do the step counts make sense?
### Sizes and steps
epoch = 0
batch_size = 320
           = 32 [num_steps/rollout] * 10 [envs]

minibatch_size = 80
               = batch_size / train_args.num_minibatches





