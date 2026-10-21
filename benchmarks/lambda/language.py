from benchmarks.common import TreeLanguage, TreeRule

from .term import Abs, App, Term, Var, free_vars, substitute


def beta(t: Term) -> Term | None:
    if isinstance(t, App) and isinstance(t.function, Abs):
        return substitute(t.function.body, t.function.parameter, t.argument)
    return None


def eta(t: Term) -> Term | None:
    if isinstance(t, Abs) and isinstance(t.body, App):
        if t.body.argument == Var(t.parameter) and t.parameter not in free_vars(t.body.function):
            return t.body.function
    return None


class LambdaLanguage(TreeLanguage[Term]):
    rules = (TreeRule("beta", beta), TreeRule("eta", eta))

    def children(self, term: Term) -> tuple[Term, ...]:
        if isinstance(term, Abs):
            return (term.body,)
        if isinstance(term, App):
            return (term.function, term.argument)
        return ()

    def rebuild(self, term: Term, children: tuple[Term, ...]) -> Term:
        if isinstance(term, Abs):
            return Abs(term.parameter, children[0])
        if isinstance(term, App):
            return App(*children)
        return term
