import torch
import numpy as np


def reaching_q(env, model, goal_zones, total_avoid_zones, value_threshold=0.85, device=torch.device('cpu')):

    model, q_net = model
    zone_index = 0
    goal_zone = goal_zones[zone_index]
    avoid_zones = total_avoid_zones[zone_index]
    env.fix_goal(goal_zone)
    ob = env.current_observation()
    zone_history = []
    task_history = []

    with torch.no_grad():
        while env.is_alive():
            confidence = False
            if avoid_zones:
                avoid_zones_ob = [env.custom_observation(goal=avoid_zone) for avoid_zone in avoid_zones]
                avoid_zones_vs = []
                for idx in range(len(avoid_zones)):
                    x_actor, x_critic = model.policy.mlp_extractor(torch.as_tensor(avoid_zones_ob[idx]).float().to(device))
                    avoid_zones_vs.append(model.policy.value_net(x_critic).detach().cpu())
                avoid_zones_vs = np.array(avoid_zones_vs)

                dangerous_zone_indices = np.argwhere(avoid_zones_vs > value_threshold)
                if dangerous_zone_indices.size > 0:

                    # NOTE: simple strategy, only avoid the most dangerous action
                    # safe_ation + goal_reaching_action (blocked) when V(avoid) > V(goal)
                    x_actor, x_critic = model.policy.mlp_extractor(torch.as_tensor(ob).float().to(device))
                    goal_v = model.policy.value_net(x_critic)
                    most_dangerous_zone_index = np.argmax(avoid_zones_vs).item()

                    if avoid_zones_vs[most_dangerous_zone_index] > goal_v:

                        # safe_action
                        # action_distribution = model.policy.get_distribution(torch.from_numpy(avoid_zones_ob[most_dangerous_zone_index]).unsqueeze(dim=0).to(device))
                        # action_probs = action_distribution.distribution.probs
                        # safe_action = torch.argmin(action_probs, dim=1)
                        
                        # Q-value version safe_action
                        # NOTE: q-values for dangerous ob
                        q_values = q_net(torch.from_numpy(avoid_zones_ob[most_dangerous_zone_index]).unsqueeze(dim=0).to(device))
                        safe_action = q_values.argmin(dim=1).reshape(-1)
                        
                        # (blocked) goal_reaching_aciton
                        # avoid_zone_action, _states = model.predict(avoid_zones_ob[most_dangerous_zone_index], deterministic=True)
                        # action_distribution = model.policy.get_distribution(torch.from_numpy(ob).unsqueeze(dim=0).to(device))
                        # action_probs = action_distribution.distribution.probs
                        # dangerous_mask = torch.ones(4).to(device)
                        # dangerous_mask[avoid_zone_action] = 0
                        # goal_reaching_action = torch.argmax(action_probs * dangerous_mask, dim=1)
                        
                        # Q-value version (blocked) goal_reaching_action
                        dangerous_action = q_values.argmax(dim=1).reshape(-1)
                        q_values = q_net(torch.from_numpy(ob).unsqueeze(dim=0).to(device))  # NOTE: q-values for ob
                        q_values[0, dangerous_action] = -float('Inf')
                        goal_reaching_action = q_values.argmax(dim=1).reshape(-1)
                        
                        ob, reward, eval_done, info = env.step({
                            'action': [safe_action, goal_reaching_action],
                            'distribution': [1, 1],
                        })
                    else:
                        confidence = True
                else:
                    confidence = True
            else:
                confidence = True
            
            if confidence:
                action, _states = model.predict(ob, deterministic=True)
                ob, reward, eval_done, info = env.step(action)

            # NOTE: entire Q
            # if confidence:
            #     q_values = q_net(torch.from_numpy(ob).unsqueeze(dim=0).to(device))
            #     action = q_values.argmax(dim=1).reshape(-1)
            #     ob, reward, eval_done, info = env.step(action)

            current_zone = info['zone']
            if current_zone:
                zone_history.append(current_zone)
            else:
                zone_history.append('+')

            if current_zone in avoid_zones:
                task_history.append(current_zone)
                print('[Dangerous !][reach {} avoid {}][overlap with {}]'.format(goal_zone, avoid_zones, info['zone']))
                return {'complete': False, 'dangerous': True, 'zone_history': zone_history, 'task_history': task_history}
            
            elif current_zone == goal_zone:
                task_history.append(current_zone)
                zone_index += 1
                if zone_index == len(goal_zones):
                    return {'complete': True, 'dangerous': False, 'zone_history': zone_history, 'task_history': task_history}
                goal_zone = goal_zones[zone_index]
                avoid_zones = total_avoid_zones[zone_index]
                env.fix_goal(goal_zones[zone_index])
                
    return {'complete': False, 'dangerous': False, 'zone_history': zone_history, 'task_history': task_history}
