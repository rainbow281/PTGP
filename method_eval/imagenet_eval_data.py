"""Read existing ImageNet-1K validation parquet, preserving official class ids."""
import hashlib,io,json
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
from torchvision import transforms

class ParquetImageNet:
    def __init__(self,path,size,indices):
        self.path=Path(path); file=pq.ParquetFile(path)
        labels=json.loads(file.schema_arrow.metadata[b'huggingface'])['info']['features']['label']['names']
        if len(labels)!=1000 or not labels[0].startswith('tench') or not labels[-1].startswith('toilet tissue'):
            raise ValueError('Unexpected ImageNet-1K label map')
        transform=transforms.Compose([transforms.Resize(size),transforms.CenterCrop(size),transforms.ToTensor()])
        self.samples={};self.manifest={'source':str(self.path.resolve()),'rows_in_shard':file.metadata.num_rows,
          'image_size':size,'preprocess':'Resize(shorter_edge)->CenterCrop->ToTensor','samples':[]}
        remaining=set(indices);offset=0
        for group in range(file.num_row_groups):
            count=file.metadata.row_group(group).num_rows
            needed=sorted(i for i in remaining if offset<=i<offset+count)
            if needed:
                table=file.read_row_group(group)
                for i in needed:
                    r=table.slice(i-offset,1).to_pylist()[0];raw=r['image']['bytes'];label=int(r['label'])
                    self.samples[i]=(transform(Image.open(io.BytesIO(raw)).convert('RGB')),label)
                    self.manifest['samples'].append(dict(shard_row=i,label=label,class_name=labels[label],
                      source_filename=r['image']['path'],image_sha256=hashlib.sha256(raw).hexdigest()))
                    remaining.remove(i)
            offset+=count
        if remaining:raise IndexError('Missing shard rows: '+str(remaining))
    def __getitem__(self,index):return self.samples[index]
