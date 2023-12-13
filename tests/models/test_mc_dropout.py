import pytest

from tests._dummies.model import dummy_model
from torch_uncertainty.models.mc_dropout import _MCDropout, mc_dropout


class TestMCDropout:
    """Testing the MC Dropout class."""

    def test_mc_dropout_train(self):
        model = dummy_model(10, 5, 1, 0.1)
        dropout_model = mc_dropout(model, num_estimators=5)
        dropout_model.train()
        assert dropout_model.training

        dropout_model = mc_dropout(model, num_estimators=5, last_layer=True)
        dropout_model.train()
        assert dropout_model.training

    def test_mc_dropout_eval(self):
        model = dummy_model(10, 5, 1, 0.1)
        dropout_model = mc_dropout(model, num_estimators=5)
        dropout_model.eval()
        assert not dropout_model.training

    def test_mc_dropout_errors(self):
        model = dummy_model(10, 5, 1, 0.1)
        num_estimators = 5

        with pytest.raises(ValueError):
            _MCDropout(model=model, num_estimators=-1, last_layer=False)

        with pytest.raises(ValueError):
            _MCDropout(model=model, num_estimators=0, last_layer=False)

        with pytest.raises(TypeError):
            dropout_model = mc_dropout(model, num_estimators)
            dropout_model.train(mode=1)

        with pytest.raises(TypeError):
            dropout_model = mc_dropout(model, num_estimators)
            dropout_model.train(mode=None)

        model = dummy_model(10, 5, 1, 0.0)
        with pytest.raises(ValueError):
            dropout_model = mc_dropout(model, num_estimators)
