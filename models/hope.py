import logging

import torch
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_rehearsal_args
from utils.buffer import Buffer
from backbone.hope import HOPEBackbone

class Hope(ContinualModel):
    NAME = 'hope'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--hope_lr', type=float, default=0.001, help='Learning rate for internal updates')
        parser.add_argument('--alpha', type=float, default=0.5, help='Legacy option; unused')
        parser.add_argument('--beta', type=float, default=0.5, help='Legacy option; unused')
        parser.add_argument('--hidden_dim', type=int, default=128, help='Legacy option; use --dim')
        parser.add_argument('--key_dim', type=int, default=64, help='Legacy option; unused')
        parser.add_argument('--memory_layers', type=int, default=3, help='Legacy option; use --depth for HOPE block count')
        parser.add_argument('--surprise_threshold', type=float,
                            help='Local update threshold (default: 1e-6 for windows, 0.15 for images)')
        parser.add_argument('--momentum', type=float, default=0.9, help='Update momentum')
        
        # Add Replay Buffer Args (optional)
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Hope, self).__init__(backbone, loss, args, transform, dataset)
        # Ensure backbone is HOPE
        if not isinstance(self.net, HOPEBackbone):
            print("Warning: HOPE model should be used with HOPE backbone. Current backbone:", type(self.net))
        
        # Optional Replay Buffer
        self.buffer = Buffer(self.args.buffer_size, self.device) if self.args.buffer_size > 0 else None

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        
        if self.buffer is not None and not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs, features = self.net(inputs, returnt='all')
        loss = self.loss(outputs, labels)
        features.retain_grad()
        loss.backward()
        
        # Local memories use the same (potentially replay-augmented) minibatch.
        # Their private vector-Jacobian products must not overwrite supervised grads.
        if isinstance(self.net, HOPEBackbone) and self.args.hope_lr != 0:
            with torch.no_grad():
                self.net(inputs, teach_signal=features.grad.detach().unsqueeze(1),
                         local_lr=self.args.hope_lr, local_momentum=self.args.momentum)
            
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        
        if self.buffer is not None:
            self.buffer.add_data(examples=not_aug_inputs, labels=labels[:real_batch_size])
        
        return loss.item()

    def end_task(self, dataset):
        """The current HOPE backbone has no task-boundary update."""
        if isinstance(self.net, HOPEBackbone):
            for i, block in enumerate(self.net.blocks):
                levels = " ".join(f"{level['name']}={m.local_updates}"
                                  for level, m in zip(block.cms.levels, block.cms.blocks))
                logging.info("HOPE cumulative local updates: block=%d clock=%d titan=%d %s",
                             i, block.cms.step_counter, block.titan.local_updates, levels)
        if hasattr(self.net, 'consolidate'):
            self.net.consolidate()
