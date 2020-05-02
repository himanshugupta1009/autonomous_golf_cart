# autonomous_golf_cart
This is our attempt at replicating the results of the famous ICRA 2015 paper on Intention aware Online POMDP planning for autonomous systems.  This project is a part of the course ASEN6519 - Decision Making Under Uncertainty that we took with Professor Zachary Sunberg in Spring 2020 at CU Boulder.

Problem Statement:

There are many successful autonomous vehicles today, but it is still uncommon to see autonomous vehicles driving among many pedestrians. To drive near pedestrians safely, efficiently, and smoothly, autonomous vehicles must estimate unknown pedestrian intentions and hedge against the uncertainty in intention estimates in order to choose actions that are effective and robust. They must also reason about the long-term effects of immediate actions.

Simple reactive control methods are inadequate here. They choose actions based on sensor data on the current system state. They do not hedge against sensing uncertainty and do not consider an action’s long-term effects. This often results in overly aggressive or conservative actions, or attractive short-term actions with undesirable longer-term consequences.


Approach:

POMDP planning provides a systematic approach to overcome these issues. The Partially Observable Markov Decision Process (POMDP) is a mathematical tool that can efficiently hedge against uncertainties to provide reliable, robust and optimal decisions or plans or control. In this work, we are reproducing the work of  <golfcart paper> for successful navigation of an autonomous cart in a dynamic environment using online POMDP planners. 


We constructed a POMDP model for autonomous driving in a dynamic environment with pedestrians. The POMDP model captures the uncertainty in pedestrian intentions and maintains a belief over pedestrian goals. In this work, we applied DESPOT, a state-of-the-art approximate online POMDP planning algorithm, to autonomous driving in a complex, dynamic environment.


We run everything in simulations on a discrete grid world with hand designed complex human motion trajectories
