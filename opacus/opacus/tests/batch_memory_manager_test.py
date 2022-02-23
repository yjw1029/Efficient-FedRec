import unittest

import torch
import torch.nn as nn
from hypothesis import given, settings
from hypothesis import strategies as st
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader, TensorDataset


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc(x)


class BatchMemoryManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data_size = 100
        self.batch_size = 10
        self.inps = torch.randn(self.data_size, 5)
        self.tgts = torch.randn(
            self.data_size,
        )

        self.dataset = TensorDataset(self.inps, self.tgts)

    def _init_training(self, **data_loader_kwargs):
        model = Model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        data_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, **data_loader_kwargs
        )

        return model, optimizer, data_loader

    @given(
        num_workers=st.integers(0, 4),
        pin_memory=st.booleans(),
    )
    @settings(deadline=10000)
    def test_basic(
        self,
        num_workers: int,
        pin_memory: bool,
    ):
        model, optimizer, data_loader = self._init_training(
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
        )
        max_physical_batch_size = 3
        with BatchMemoryManager(
            data_loader=data_loader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer,
        ) as new_data_loader:
            self.assertEqual(
                len(data_loader), len(data_loader.dataset) // self.batch_size
            )
            self.assertEqual(
                len(new_data_loader),
                len(data_loader.dataset) // max_physical_batch_size,
            )
            weights_before = torch.clone(model._module.fc.weight)
            for i, (x, y) in enumerate(new_data_loader):
                self.assertTrue(x.shape[0] <= 3)

                out = model(x)
                loss = (y - out).mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if i % 4 < 3:
                    self.assertTrue(
                        torch.allclose(model._module.fc.weight, weights_before)
                    )
                else:
                    self.assertFalse(
                        torch.allclose(model._module.fc.weight, weights_before)
                    )
                    weights_before = torch.clone(model._module.fc.weight)

    def test_equivalent_to_one_batch(self):
        torch.manual_seed(1337)
        model, optimizer, data_loader = self._init_training()

        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
        )

        with BatchMemoryManager(
            data_loader=data_loader, max_physical_batch_size=3, optimizer=optimizer
        ) as data_loader:
            for x, y in data_loader:
                out = model(x)
                loss = (y - out).mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        memory_manager_weights = model._module.fc.weight.detach()

        torch.manual_seed(1337)
        model, optimizer, data_loader = self._init_training()

        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
        )

        for x, y in data_loader:
            out = model(x)
            loss = (y - out).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        vanilla_weights = model._module.fc.weight.detach()

        self.assertTrue(torch.allclose(memory_manager_weights, vanilla_weights))
