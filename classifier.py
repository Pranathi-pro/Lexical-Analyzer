import re
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

KEYWORDS = {
    "int", "float", "double", "char", "bool", "void", "if", "else", "while",
    "for", "return", "class", "public", "private", "static", "def", "print",
    "import", "from", "True", "False", "break", "continue"
}
TOKEN_RE = re.compile(
    r'"(?:\\.|[^"\\])*"|\\\'(?:\\\\.|[^\\\'\\\\])*\\\'|'
    r'[A-Za-z_][A-Za-z0-9_]*|\\d+(?:\\.\\d+)?|'
    r'==|!=|<=|>=|\\+\\+|--|&&|\\|\\||'
    r'[+\\-*/%=<>!&|^~]|[(){}\\[\\],.;:]'
)

def basic_type(token):
    if token in KEYWORDS:
        return "keyword"
    if re.fullmatch(r'\\d+(?:\\.\\d+)?', token):
        return "number"
    if token.startswith('"') or token.startswith("'"):
        return "string"
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', token):
        return "identifier"
    if token in "+-*/%=<>!&|^~":
        return "operator"
    return "punctuation"

class ContextAwareTokenClassifier:
    """Small supervised ML demo using token and neighboring-token features."""
    def __init__(self):
        examples = [
            ("int", "total", "=", "keyword"), ("float", "rate", ";", "keyword"),
            ("if", "(", "x", "keyword"), ("return", "value", ";", "keyword"),
            ("x", "=", "42", "identifier"), ("total", "+", "price", "identifier"),
            ("25", ";", "}", "number"), ("3.14", "+", "x", "number"),
            ('"hello"', ";", "}", "string"), ("'a'", ";", "}", "string"),
            ("=", "x", ";", "operator"), ("+", "2", ";", "operator"),
            ("(", "x", ")", "punctuation"), (";", "}", "}", "punctuation"),
            ("print", "(", "x", "keyword"), ("class", "A", "{", "keyword"),
            ("name", "=", '"Sam"', "identifier"), ("while", "(", "i", "keyword"),
            ("i", "<", "10", "identifier"), ("<", "10", ")", "operator"),
        ]
        X, y = [], []
        # Repeat and vary neighboring context to create a modest demonstration set.
        contexts = [(";", "}"), ("(", ")"), ("=", ";"), ("+", ";"), ("{", "}")]
        for token, prev, nxt, label in examples:
            X.append(self._features(token, prev, nxt))
            y.append(label)
            for p, n in contexts:
                X.append(self._features(token, p, n))
                y.append(label)
        self.pipeline = Pipeline([
            ("vectorizer", DictVectorizer(sparse=False)),
            ("classifier", DecisionTreeClassifier(max_depth=8, random_state=7))
        ])
        self.pipeline.fit(X, y)

    @staticmethod
    def _features(token, prev="", nxt=""):
        return {
            "token": token,
            "basic_type": basic_type(token),
            "previous_token": prev,
            "next_token": nxt,
            "starts_upper": str(token[:1].isupper()),
            "has_digit": str(any(c.isdigit() for c in token)),
            "length_bucket": "short" if len(token) <= 3 else "long",
        }

    def analyze(self, source):
        raw = TOKEN_RE.findall(source)
        rows = []
        for i, token in enumerate(raw):
            prev = raw[i - 1] if i else "<START>"
            nxt = raw[i + 1] if i + 1 < len(raw) else "<END>"
            features = self._features(token, prev, nxt)
            prediction = self.pipeline.predict([features])[0]
            rows.append({
                "Token": token,
                "Previous": prev,
                "Next": nxt,
                "Rule-based Type": basic_type(token),
                "Predicted Class": prediction
            })
        return rows
