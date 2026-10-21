"""Synthetic CPU checks only; no dataset or checkpoint access."""
import copy
import unittest
import torch
from opsa import OPSAAdapter, quadrature


class OPSATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(74430)
        torch.set_num_threads(1)

    def test_quadrature_and_radius(self):
        a = torch.randn(3, 7, 16, dtype=torch.double)
        p = quadrature(a)
        torch.testing.assert_close(quadrature(p), -a)
        torch.testing.assert_close(p.norm(dim=-1), a.norm(dim=-1))
        self.assertLess((a * p).sum(-1).abs().max().item(), 1e-12)
        m = OPSAAdapter(8, 4).double()
        self.assertAlmostEqual(m.effective_up().norm().item(), 8)
        self.assertEqual(sum(p.numel() for p in m.parameters()), 2 * 8 * 4)

    def test_forward_reference_and_gradients(self):
        m = OPSAAdapter(8, 4).double()
        x = torch.randn(2, 3, 8, dtype=torch.double, requires_grad=True)
        j = torch.tensor([[0., -1., 0., 0.], [1., 0., 0., 0.],
                          [0., 0., 0., -1.], [0., 0., 1., 0.]], dtype=torch.double)
        a = (x / x.norm(dim=-1, keepdim=True) * 8 ** .5) @ m.down.T
        p = a @ j.T
        expected = x + (a * p.cos() + p * a.cos()) @ m.effective_up().T
        torch.testing.assert_close(m(x), expected)
        self.assertTrue(torch.autograd.gradcheck(m, (x,)))
        m(x).square().mean().backward()
        for param in m.parameters():
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(param.grad.norm().item(), 0)
        other = OPSAAdapter(8, 4).double()
        other.load_state_dict(copy.deepcopy(m.state_dict()))
        torch.testing.assert_close(other(x), m(x), rtol=0, atol=0)

    def test_mapc_reset_and_adam_accumulation(self):
        m = OPSAAdapter(2, 2).double()
        opt = torch.optim.AdamW(m.parameters(), lr=.001)
        x = torch.tensor([[1., 1.]], dtype=torch.double)
        m(x, collect_mapc=True).sum().backward()
        self.assertEqual(m.reset_mapc(opt), 0)
        opt.step()
        opt.zero_grad()
        with torch.no_grad():
            m.down.copy_(torch.tensor([[2., 0.], [0., 0.]], dtype=torch.double))
        m(x, collect_mapc=True).sum().backward()
        m.down.grad.copy_(torch.tensor([[-2., 2.], [-2., 2.]], dtype=torch.double))
        moment = opt.state[m.down]['exp_avg']
        moment.fill_(1.)
        second = opt.state[m.down]['exp_avg_sq'].clone()
        self.assertEqual(m.reset_mapc(opt), 1)
        torch.testing.assert_close(moment, torch.tensor([[0., 1.], [1., 1.]], dtype=torch.double))
        torch.testing.assert_close(opt.state[m.down]['exp_avg_sq'], second, rtol=0, atol=0)
        opt.step()
        self.assertAlmostEqual(moment[0, 0].item(), -.2)
        self.assertAlmostEqual(moment[1, 0].item(), .7)
        self.assertFalse(m.boundary_rows.any())
        with self.assertRaises(RuntimeError):
            m.reset_mapc(opt)

    def test_fail_closed(self):
        with self.assertRaises(ValueError):
            OPSAAdapter(8, 3)
        m = OPSAAdapter(8, 4)
        with self.assertRaises(ValueError):
            m(torch.zeros(2, 8))
        m.eval()
        with self.assertRaises(ValueError):
            m(torch.ones(2, 8), collect_mapc=True)
        with torch.no_grad():
            m.up.zero_()
        with self.assertRaises(ValueError):
            m(torch.ones(2, 8))

    def test_microbatch_union_and_nonfinite_gradient(self):
        m = OPSAAdapter(2, 2).double()
        opt = torch.optim.AdamW(m.parameters())
        with torch.no_grad():
            m.down.copy_(torch.eye(2, dtype=torch.double) * 2)
        m(torch.tensor([[1., 0.]], dtype=torch.double), collect_mapc=True).sum().backward()
        m(torch.tensor([[0., 1.]], dtype=torch.double), collect_mapc=True).sum().backward()
        self.assertTrue(m.boundary_rows.all())
        m.up.grad[0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            m.reset_mapc(opt)
        self.assertFalse(m.boundary_rows.any())


if __name__ == '__main__':
    unittest.main()
