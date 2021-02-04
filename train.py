import pandas as pd
import glob
import os

import rdkit
from tqdm import trange, tqdm


import os
import os.path as osp
import re

import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_gz)

from torch_geometric.data import DataLoader

import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.data import DataLoader
from torch_geometric.nn import BatchNorm, global_mean_pool

from models.pytorch_geometric.pna import PNAConvSimple


try:
    from rdkit import Chem
except ImportError:
    Chem = None

x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}


def main():

    train_df = pd.read_csv('train/raw/train.csv')
    test_df = pd.read_csv('test/raw/holdout_set.csv')
    # eval_df = pd.read_csv('alt/raw/Lipophilicity.csv', usecols=['exp','smiles']).rename(columns={'smiles':'Smiles','exp': 'label'})

    df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', "./alt/raw/*.csv"))))
    df=df[['smiles','logP','logD']]
    # df = df.drop_duplicates('smiles')
    df['label'] = df['logD']
    df['label'] = df['label'].fillna(df['logP'])
    df = df.drop(columns=['logP','logD']).rename(columns = {'smiles':'Smiles'})
    augmented_df = pd.concat((train_df,df)).dropna().drop_duplicates('Smiles')
    augmented_df.to_csv('/content/molecular-GNN/train_augmented/raw/train_augmented.csv',index=False)

    torch.manual_seed(12345)

    train_dataset = MoleculeNet(root='./',name='train_augmented')
    train_dataset.process()

    test_dataset = MoleculeNet(root='./',name='test')
    test_dataset.process()

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    # print(f'Number of eval graphs: {len(eval_dataset)}')

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    train_loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    for step, data in enumerate(test_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, min_lr=0.0001)


    # Compute in-degree histogram over training data.

    deg = torch.zeros(10, dtype=torch.long)
    for data in tqdm(train_dataset):
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())


    for epoch in range(1, 201):
        loss = train(epoch) #MAE
        print(f'Epoch: {epoch:02d}, MAE Loss: {loss:.4f}')

    out = [float(model(data.x, data.edge_index, None, data.batch).cpu().detach().numpy().squeeze()) for data in test_loader]

    submission_df = pd.read_csv('test/raw/holdout_set.csv')
    submission_df['predicted']=out
    submission_df.to_csv('submissions/holdout_set.csv',index=False)


def train(epoch):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, None, data.batch)

        loss = torch.nn.L1Loss()(out.to(torch.float32), data.y.to(torch.float32))
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


class MoleculeNet(InMemoryDataset):
    r"""The `MoleculeNet <http://moleculenet.ai/datasets-1>`_ benchmark
    collection  from the `"MoleculeNet: A Benchmark for Molecular Machine
    Learning" <https://arxiv.org/abs/1703.00564>`_ paper, containing datasets
    from physical chemistry, biophysics and physiology.
    All datasets come with the additional node and edge features introduced by
    the `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"ESOL"`,
            :obj:`"FreeSolv"`, :obj:`"Lipo"`, :obj:`"PCBA"`, :obj:`"MUV"`,
            :obj:`"HIV"`, :obj:`"BACE"`, :obj:`"BBPB"`, :obj:`"Tox21"`,
            :obj:`"ToxCast"`, :obj:`"SIDER"`, :obj:`"ClinTox"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    # Format: name: [display_name, url_name, csv_name, smiles_idx, y_idx]
    names = {
        'train': ['Train', '60170fff5b630f3433c202be_train.csv', '60170fff5b630f3433c202be_train', 0, 1],
        'test':['Test','60170fffa2720fa4d0b9067a_holdout_set.csv','60170fffa2720fa4d0b9067a_holdout_set',0,1],
        'eval': ['Lipophilicity Filtered', 'Lipophilicity_filtered.csv', 'Lipophilicity_filtered', 0, 1], #stanford dataset
        'train_augmented':['Train Augmented','train_augmented.csv','train_augmented',0,1], #vantai train + stanford
    }

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):

        if Chem is None:
            raise ImportError('`MoleculeNet` requires `rdkit`.')

        self.name = name.lower()
        assert self.name in self.names.keys()
        super(MoleculeNet, self).__init__(root, transform, pre_transform,
                                          pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.names[self.name][2]}.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            # dataset = f.read().split('\n')[1:-1]

            dataset = f.read().split('\n')[1:] #issue w/ test loader
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for line in dataset:
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            line = line.split(',')

            smiles = line[self.names[self.name][3]]
            ys = line[self.names[self.name][4]]
            ys = ys if isinstance(ys, list) else [ys]

            ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            xs = []
            for atom in mol.GetAtoms():
                x = []
                x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
                x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
                x.append(x_map['degree'].index(atom.GetTotalDegree()))
                x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
                x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
                x.append(x_map['num_radical_electrons'].index(
                    atom.GetNumRadicalElectrons()))
                x.append(x_map['hybridization'].index(
                    str(atom.GetHybridization())))
                x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
                x.append(x_map['is_in_ring'].index(atom.IsInRing()))
                xs.append(x)

            x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                e = []
                e.append(e_map['bond_type'].index(str(bond.GetBondType())))
                e.append(e_map['stereo'].index(str(bond.GetStereo())))
                e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

                edge_indices += [[i, j], [j, i]]
                edge_attrs += [e, e]

            edge_index = torch.tensor(edge_indices)
            edge_index = edge_index.t().to(torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                        smiles=smiles)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.node_emb = AtomEncoder(emb_dim=70)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConvSimple(in_channels=70, out_channels=70, aggregators=aggregators,
                                 scalers=scalers, deg=deg, post_layers=1)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(70))

        self.mlp = Sequential(Linear(70, 35), ReLU(), Linear(35, 17), ReLU(), Linear(17, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            h = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
            x = h + x  # residual#
            x = F.dropout(x, 0.3, training=self.training)

        x = global_mean_pool(x, batch)
        return self.mlp(x)



if __name__=='__main__':
    main()