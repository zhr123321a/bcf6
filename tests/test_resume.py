"""Check continuation against uninterrupted optimizer/RNG trajectories."""
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from bcf6.model import BCF6
from bcf6.train import checkpoint,make_optimizer


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_checkpoint_continues_all_stages(self):
        def update(model,opt,scheduler):
            uv=torch.rand(32,2);scale=random.uniform(0,2)
            target=torch.cat([uv,uv,uv,uv],-1)
            opt.zero_grad();loss=(model(uv,scale)-target).square().mean()
            loss.backward();opt.step();model.constrain();scheduler.step()
            return loss.detach()

        for stage in ('warmup','block','finetune'):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as temp:
                torch.manual_seed(19);random.seed(19)
                model=BCF6('0.5K',8,(32,32),sides=[16,8,4,4])
                if stage!='warmup':model=model.to_blocks()
                if stage=='finetune':model.freeze_quantized()
                opt,scheduler=make_optimizer(model,stage)
                for _ in range(3):update(model,opt,scheduler)
                path=Path(temp)/'checkpoint.pt'
                checkpoint(path,model,opt,scheduler,stage,3,3,SimpleNamespace(seed=19),{'target_sha256':'synthetic'})
                expected_loss=update(model,opt,scheduler)
                expected={k:v.clone() for k,v in model.state_dict().items()}
                saved=torch.load(path,map_location='cpu',weights_only=False)
                restored=BCF6.from_state(saved['model_config'],saved['model'])
                opt2,scheduler2=make_optimizer(restored,stage)
                opt2.load_state_dict(saved['optimizer']);scheduler2.load_state_dict(saved['scheduler'])
                torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng'])
                actual_loss=update(restored,opt2,scheduler2)
                torch.testing.assert_close(actual_loss,expected_loss,rtol=0,atol=0)
                for name,value in restored.state_dict().items():
                    torch.testing.assert_close(value,expected[name],rtol=0,atol=0,msg=name)
                self.assertEqual(scheduler.state_dict(),scheduler2.state_dict())


if __name__=='__main__':unittest.main()
