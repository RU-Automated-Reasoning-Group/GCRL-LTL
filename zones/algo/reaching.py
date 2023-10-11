import torch
import numpy as np


def reaching(env, model, goal_zones, total_avoid_zones, value_threshold=0.85, device=torch.device('cpu')):

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
                        action_distribution = model.policy.get_distribution(torch.from_numpy(avoid_zones_ob[most_dangerous_zone_index]).unsqueeze(dim=0).to(device))
                        action_probs = action_distribution.distribution.probs
                        safe_action = torch.argmin(action_probs, dim=1)

                        # (blocked) goal_reaching_aciton
                        avoid_zone_action, _states = model.predict(avoid_zones_ob[most_dangerous_zone_index], deterministic=True)
                        action_distribution = model.policy.get_distribution(torch.from_numpy(ob).unsqueeze(dim=0).to(device))
                        action_probs = action_distribution.distribution.probs
                        dangerous_mask = torch.ones(4).to(device)
                        dangerous_mask[avoid_zone_action] = 0
                        goal_reaching_action = torch.argmax(action_probs * dangerous_mask, dim=1)

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


def gc_reaching(env, policy, gcvf, goal_zones, total_avoid_zones, value_threshold=0.85, device=torch.device('cpu')):

    zone_index = 0
    goal_zone = goal_zones[zone_index]
    avoid_zones = total_avoid_zones[zone_index]
    env.fix_goal(goal_zone)
    ob = env.current_observation()
    zone_history = []
    task_history = []

    # from PIL import Image
    # image = Image.fromarray(env.render(mode='rgb_array', camera_id=1, height=400, width=400))
    # image.save('start.png')

    import copy
    anywhere_ob = env.goals_representation['ANYWHERE']
    red_ob = env.goals_representation['R']
    jetblack_ob = env.goals_representation['J']
    white_ob = env.goals_representation['W']
    yellow_ob = env.goals_representation['Y']

    with torch.no_grad():
        while env.is_alive():

            # DEBUG
            _ob = copy.deepcopy(ob)
            _ob[-24:] = white_ob

            #print('[GoalValue] [ ANYWHERE] -> [GOAL]', gcvf.predict(np.concatenate((anywhere_ob, _ob))))
            #print('[GoalValue] [JET BLACK] -> [GOAL]', gcvf.predict(np.concatenate((jetblack_ob, _ob))))
            #print('[GoalValue] [    WHITE] -> [GOAL]', gcvf.predict(np.concatenate((white_ob, _ob))))
            #print('[GoalValue] [      RED] -> [GOAL]', gcvf.predict(np.concatenate((red_ob, _ob))))
            print('[GoalValue] [   YELLOW] -> [GOAL]', gcvf.predict(np.concatenate((yellow_ob, _ob))))

            confidence = False
            if avoid_zones:
                avoid_zones_ob = [env.custom_observation(goal=avoid_zone) for avoid_zone in avoid_zones]
                avoid_zones_vs = []
                for idx in range(len(avoid_zones)):
                    x_actor, x_critic = policy.mlp_extractor(torch.as_tensor(avoid_zones_ob[idx]).float().to(device))
                    avoid_zones_vs.append(policy.value_net(x_critic).detach().cpu())
                avoid_zones_vs = np.array(avoid_zones_vs)

                dangerous_zone_indices = np.argwhere(avoid_zones_vs > value_threshold)
                if dangerous_zone_indices.size > 0:

                    # NOTE: simple strategy, only avoid the most dangerous action
                    # safe_ation + goal_reaching_action (blocked) when V(avoid) > V(goal)
                    x_actor, x_critic = policy.mlp_extractor(torch.as_tensor(ob).float().to(device))
                    goal_v = policy.value_net(x_critic)
                    most_dangerous_zone_index = np.argmax(avoid_zones_vs).item()

                    if avoid_zones_vs[most_dangerous_zone_index] > goal_v:

                        # safe_action
                        action_distribution = policy.get_distribution(torch.from_numpy(avoid_zones_ob[most_dangerous_zone_index]).unsqueeze(dim=0).to(device))
                        action_probs = action_distribution.distribution.probs
                        safe_action = torch.argmin(action_probs, dim=1)

                        # (blocked) goal_reaching_aciton
                        avoid_zone_action, _states = policy.predict(avoid_zones_ob[most_dangerous_zone_index], deterministic=True)
                        action_distribution = policy.get_distribution(torch.from_numpy(ob).unsqueeze(dim=0).to(device))
                        action_probs = action_distribution.distribution.probs
                        dangerous_mask = torch.ones(4).to(device)
                        dangerous_mask[avoid_zone_action] = 0
                        goal_reaching_action = torch.argmax(action_probs * dangerous_mask, dim=1)

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
                action, _states = policy.predict(ob, deterministic=True)
                ob, reward, eval_done, info = env.step(action)

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
