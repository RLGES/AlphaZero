from random import Random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from benchmarks.generators import generate
VALIDATION_SEED = 515_151
def run(job):
    term, policy = job
    from benchmarks.asm.geb import Agent, EGraphEnv, SearchConfig, build, egglog_schedule
    from benchmarks.asm.language import AsmLanguage
    from benchmarks.asm.final_eval import _network
    lang = AsmLanguage(); env = EGraphEnv(lang, node_limit=1000, max_steps=8, kind="enable")
    if policy == "egglog":
        ep = egglog_schedule(term, env)
    elif policy == "builtin-only":
        env.reset(term); best = env.extract()[1]
        while not env.terminal():
            env.apply(env.noop); best = min(best, env.extract()[1])
        return 100*(env.initial_cost-best)/env.initial_cost, env.size()
    elif policy == "all-enabled":
        env.reset(term); best = env.extract()[1]
        for a in env.valid: 
            if a != env.noop: env.active += (a,)
        while not env.terminal():
            env.apply(env.noop); best = min(best, env.extract()[1])
        return 100*(env.initial_cost-best)/env.initial_cost, env.size()
    else:
        net = _network(policy)
        ep = build(term, env, Agent(lang, net), SearchConfig(simulations=16, mode="az"), Random(0))
    return 100*(ep.initial_cost-ep.best_cost)/ep.initial_cost, ep.final_size
if __name__ == "__main__":
    terms = generate("asm", Random(VALIDATION_SEED), 40, (4, 6), split="train")
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(4, mp_context=ctx) as pool:
        for policy in ["egglog", "builtin-only", "all-enabled", "checkpoints/geb_v1/best.pt"]:
            r = list(pool.map(run, [(t, policy) for t in terms]))
            print(policy, round(sum(x[0] for x in r)/40, 2), "nodes", int(sum(x[1] for x in r)/40), flush=True)
