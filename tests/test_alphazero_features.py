from benchmarks.arithmetic import Add, ArithmeticLanguage, Const, Mul, Var
from benchmarks.bitvector import Const as BVConst
from benchmarks.registry import SUITE_NAMES, load_suite
from src import Language
from src.engines.alphazero.features import GraphFeatures, supports_alphazero


class PlainLanguage(Language[int, str]):
    def applicable(self, term):
        return ()

    def apply(self, term, action):
        return term

    def cost(self, term):
        return 0.0

    def fingerprint(self, term):
        return term


def test_tree_features_preorder_and_positions():
    x = Var("x")
    features = ArithmeticLanguage().features(Add(Mul(x, Const(1)), Const(0)))
    assert features.labels == ("add", "mul", "var", "const:1", "const:0")
    assert features.edges == ((0, 1, 0), (1, 2, 0), (1, 3, 1), (0, 4, 1))
    assert features.root == 0


def test_action_vocabulary_is_rules_and_contains_applicable():
    for name in SUITE_NAMES:
        suite = load_suite(name)
        language = suite.language
        assert language.action_vocabulary() == language.rules
        assert supports_alphazero(language)
        for benchmark in suite.benchmarks:
            assert set(language.applicable(benchmark.term)) <= set(language.action_vocabulary())
            features = language.features(benchmark.term)
            assert isinstance(features, GraphFeatures)
            assert len(features.labels) == language.size(benchmark.term)
            assert len(features.edges) == len(features.labels) - 1


def test_big_constants_bucketed_and_variable_names_dropped():
    language = ArithmeticLanguage()
    assert language.node_label(Const(1000)) == "const:big"
    assert language.node_label(Const(-3)) == "const:-3"
    assert language.node_label(Var("anything")) == "var"


def test_suite_specific_labels():
    bitvector = load_suite("bitvector").language
    assert bitvector.node_label(BVConst(255)) == "const:ones"
    assert bitvector.node_label(BVConst(0)) == "const:0"
    assert bitvector.node_label(BVConst(7)) == "const:other"
    linear = load_suite("linear_algebra")
    label = linear.language.node_label(linear.benchmarks[0].term)
    assert label.count("x") == 1 and ":" in label


def test_supports_alphazero_false_for_plain_language():
    assert not supports_alphazero(PlainLanguage())
