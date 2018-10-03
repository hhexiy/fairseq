import numpy as np
import torch

from . import data_utils, LanguagePairDataset, IndexedInMemoryDataset

class EditDatasetSrcWrapper(object):
    def __init__(self, ds):
        self.ds = ds
        self.size = ds.size // 3
        self.sizes = np.array([ds.sizes[i] for i in range(2, len(ds), 3)])

    def __getitem__(self, index):
        return {
                'deleted': self.ds[index * 3 + 0],
                'related': self.ds[index * 3 + 1],
                'template': self.ds[index * 3 + 2],
                }

    def __len__(self):
        return len(self.ds) // 3

def collate(samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False, insert='none'):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source-template', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source-template'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    src_insert = None
    if insert != 'none':
        src_insert = merge('source-insert', left_pad=left_pad_source)
        src_insert = src_insert.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge(
            'target',
            left_pad=left_pad_target,
            move_eos_to_beginning=True,
        )
        prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)
    else:
        ntokens = sum(len(s['source-template']) for s in samples)

    item = {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'prev_output_tokens': prev_output_tokens,
        },
        'target': target,
    }
    if insert != 'none':
        item['net_input']['src_insert'] = src_insert
    return item



class EditDataset(LanguagePairDataset):
    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True,
        insert='none', combine='embedding',
    ):
        super().__init__(
            src, src_sizes, src_dict,
            tgt=tgt, tgt_sizes=tgt_sizes, tgt_dict=tgt_dict,
            left_pad_source=left_pad_source, left_pad_target=left_pad_target,
            max_source_positions=max_source_positions, max_target_positions=max_target_positions,
            shuffle=shuffle,
            )
        self.insert = insert
        self.combine = combine

    def __getitem__(self, index):
        item = {
            'id': index,
            'source-template': self.src[index]['template'],
            'target': self.tgt[index] if self.tgt is not None else None,
        }
        if self.insert == 'deleted':
            item['source-insert'] = self.src[index]['deleted']
        elif self.insert == 'related':
            item['source-insert'] = self.src[index]['related']
        if self.combine == 'token' and self.insert != 'none':
            # TODO: use different seperator
            template = torch.cat((item['source-template'], item['source-insert'], torch.LongTensor([self.src_dict.eos()])), dim=0)
            #print(item['source-template'])
            #print(item['source-template'].size())
            #print(item['source-insert'])
            #print(item['source-insert'].size())
            #eos = self.src_dict.eos()
            #print(eos)
            #print(torch.LongTensor([eos]).size())
            #print(torch.LongTensor([eos]))
            #print(template.size())
            #import sys; sys.exit()
            item['source-template'] = template
        return item

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        insert = 'none' if self.combine == 'token' else self.insert
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            insert=insert,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        src_len, tgt_len = min(src_len, max_source_positions), min(tgt_len, max_target_positions)
        bsz = num_tokens // max(src_len, tgt_len)
        return self.collater([
            {
                'id': i,
                'source-template': self.src_dict.dummy_sentence(src_len),
                'source-insert': self.src_dict.dummy_sentence(1),
                'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
            }
            for i in range(bsz)
        ])



# test
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    args = parser.parse_args()

    src_data = '{}/train.src-tgt.src'.format(args.data)
    src_dict = '{}/dict.src.txt'.format(args.data)
    tgt_data = '{}/train.src-tgt.tgt'.format(args.data)
    tgt_dict = '{}/dict.tgt.txt'.format(args.data)

    def indexed_dataset(path, dictionary):
        if IndexedInMemoryDataset.exists(path):
            return IndexedInMemoryDataset(path, fix_lua_indexing=True)
        return None

    src_dataset = indexed_dataset(src_data, src_dict)
    wrapped_src_dataset = SrcDatasetWrapper(indexed_dataset(src_data, src_dict))
    tgt_dataset = indexed_dataset(tgt_data, tgt_dict)
    print(src_dataset.size)
    print(src_dataset.sizes)
    print(src_dataset.sizes.dtype)
    print(wrapped_src_dataset.size)
    print(wrapped_src_dataset.sizes)
    print(wrapped_src_dataset.sizes.dtype)
    print(tgt_dataset.size)
    print(tgt_dataset.sizes)
