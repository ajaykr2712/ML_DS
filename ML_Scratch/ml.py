from packaging import version
import sklearn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
