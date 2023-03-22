## Misc TODO
1. Use tqdm for training loop
2. Manual_Test_overide

## Seeding
1. Need to figure out how to seed gym environments properly in addition to all DRL code

# Value Estimation

## Advantage Formula

## Value Function Bootstrapping
"If a sub-environment is not terminated nor truncated, PPO estimates the value of the next state in this sub-environment as the value target." [37]

We are technically truncating the environment because we are wrapping it in a time-limit wrapper. I think this means we are NOT bootstrapping


## Value Function Loss
Unclipped VF loss is

$$(v_\theta - v_{target})^2$$
