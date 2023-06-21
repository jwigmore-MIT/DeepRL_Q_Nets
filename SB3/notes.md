
**Intervention Aided Policy Seems to be the way to go**
But should we use PPO or a deterministic policy gradient algorithm?
Lets explore both options.

# Unanswered Questions

### Do we want to normalize the observation and rewards?
Pros: helps during early stages of training for value function estimation
Cons: Once the policy starts to converge to a good policy, the state distribution will change during the intermediate term and eventually it will settle out

We also have the challenge of dealing with normalization across different environments during training and testing...
Quick solution: Truncation i.e. restrict observations/rewards to be within a certain range and normalize by the range. 

If we are continuously updating the reward normalization factors, then this will depend on the current environment dynamics. 

### Should the policy converge to zero std?
The standard deviation of the policy is essentially the noise parameter shared between all actions.  If the policy converges to zero std this means we have a deterministic policy.

### What is taking so long for wandb to upload?
Might be tensorboard logging.  Try disabling it.


___
# Test Logs

## State Normalization
What: Remove state normalization and compare the convergence rate of the policy and value function
- Also monitor testing performance.  How does state normalization work when we are testing on a different environment than training?

## Reward Normalization
