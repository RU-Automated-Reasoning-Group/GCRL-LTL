import numpy as np


def monitor_predicate(predicate, state_dim):
    '''
    Convert a system-resource predicate to a monitor predicate, i.e., over S*V.
    '''
    def new_predicate(state, reg):
        pvalue = predicate(state[:state_dim], state[state_dim:])
        return pvalue > 0, pvalue
    return new_predicate


def true_predicate(local_reward_bound):
    '''
    True Predicate with quantitative semantics determined by the local reward bound.
    '''
    def predicate(state, reg):
        return (True, local_reward_bound/2)
    return predicate


def project_predicate(mpred, s_reg, e_reg):
    '''
    Project predicate to a sub list of register values.
    s_reg: starting register number
    e_reg: ending register number
    '''
    def predicate(state, reg):
        return mpred(state, reg[s_reg:e_reg])
    return predicate


def project_update(mupdate, s_reg, e_reg, clean=False):
    '''
    Project update to a sub list of registers.
    '''
    def update(state, reg):
        retval = reg.copy()
        if clean:
            retval = np.zeros(len(reg))
        retval[s_reg:e_reg] = mupdate(state, reg[s_reg:e_reg])
        return retval
    return update


def project_reward(mrew, s_reg, e_reg):
    '''
    Project reward to a sub list of registers.
    '''
    def rew(state, reg):
        return mrew(state, reg[s_reg:e_reg])
    return rew


def alw_reward(mrew):
    '''
    Reward for alw construction.
    '''
    def rew(state, reg):
        return min(mrew(state, reg[:-1]), reg[-1])
    return rew


def seq_reward(mrew, reg_no):
    '''
    Reward for seq construction.
    '''
    def rew(state, reg):
        return min(mrew(state, reg[:reg_no]), reg[-1])
    return rew


def rew_pred(mpred, mrew, reg_init, s_reg, e_reg):
    '''
    Combine predicate with reward positivity condition.
    reg_init: array (initial register for the second monitor)
    '''
    def predicate(state, reg):
        (rb, rv) = mpred(state, reg_init)
        rew = mrew(state, reg[s_reg:e_reg])
        return (rb and rew > 0, min(rv, rew))
    return predicate


def conj_pred(mpred1, mpred2, reg_init):
    '''
    Conjunction with base predicate.
    '''
    def predicate(state, reg):
        (b1, v1) = mpred1(state, reg_init)
        (b2, v2) = mpred2(state, reg)
        return (b1 and b2, min(v1, v2))
    return predicate


def neg_pred(mpred):
    '''
    Negation of a predicate.
    '''
    def predicate(state, reg):
        (b, v) = mpred(state, reg)
        return (not b, -v)
    return predicate


def init_update(mupdate, reg_init):
    '''
    Update based on initial register value.
    '''
    def update(state, reg):
        retval = reg.copy()
        retval[:len(reg_init)] = mupdate(state, reg_init)
        return retval
    return update


def alw_update(mupdate, mpred):
    '''
    Change update to track satisfaction of safety constraints.
    '''
    def update(state, reg):
        return np.concatenate([mupdate(state, reg[:-1]), np.array([min(reg[-1],
                                                                       mpred(state, reg)[1])])])
    return update


def seq_update(total_reg_no, mon1_reg_no, mon2_reg_no, mon2_init_reg, mon1_rew, mupdate):
    '''
    Update function for sequence.
    '''
    def update(state, reg):
        retval = np.zeros(total_reg_no)
        retval[:mon2_reg_no] = mupdate(state, mon2_init_reg)
        retval[-1] = mon1_rew(state, reg[:mon1_reg_no])
        return retval
    return update


def id_update(state, reg):
    '''
    Identity update.
    '''
    return reg.copy()
